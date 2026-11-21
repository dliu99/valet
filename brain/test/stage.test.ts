import assert from "node:assert/strict";
import { test } from "node:test";
import { askTask, runTask } from "../stage.ts";
import type { Task } from "../task.ts";
import { choose, rig, scriptedDecider } from "./helpers.ts";

const go = () => new AbortController().signal;
const task: Task = { id: "0", goal: "Pick up the bottle.", seconds: 30 };

test("Jev's choice and distance are carried out, and the next tick sees the result", async () => {
	const r = await rig({ decider: () => scriptedDecider((req, tick) => choose(req, tick! < 3 ? "reach_out" : "stuck", { far: 2 })) });
	try {
		const report = await runTask(task, r.deps, go());
		assert.equal(report.exit, "stuck");
		const motions = r.sink.records.filter((x) => x.kind === "valet.motion").map((m) => m.data.motion);
		assert.deepEqual(motions, [{ kind: "delta", reach_m: 0.02 }, { kind: "delta", reach_m: 0.02 }]);
		assert.deepEqual(report.steps[0], { tick: 1, action: "reach_out", p: 0.9, step: "2.0 cm", did: "moved" });
		const second = r.sink.records.filter((x) => x.kind === "valet.tick")[1]!.data.state as { recent_actions: unknown[] };
		assert.deepEqual(second.recent_actions, [{ action: "reach_out", step: "2.0 cm", result: "moved" }]);
	} finally {
		await r.close();
	}
});

test("done counts when Jev chooses it on two ticks in a row", async () => {
	const r = await rig({ decider: () => scriptedDecider((req, tick) => choose(req, tick === 2 ? "lower" : "done")) });
	try {
		const report = await runTask(task, r.deps, go());
		assert.equal(report.exit, "done");
		assert.deepEqual(report.steps.map((s) => s.action), ["done", "lower", "done", "done"]);
		assert.equal(report.steps[0]!.did, "claimed done; looking again");
	} finally {
		await r.close();
	}
});

test("a refused move is a result Jev sees, not the end", async () => {
	const r = await rig({ maxReachM: 0.14, decider: () => scriptedDecider((req, tick) => choose(req, tick! < 3 ? "reach_out" : "stuck", { far: 3 })) });
	try {
		const report = await runTask(task, r.deps, go());
		assert.equal(report.exit, "stuck");
		assert.match(report.steps[0]!.did, /^refused: out_of_envelope/);
		assert.match(report.steps[1]!.did, /^refused/);
	} finally {
		await r.close();
	}
});

test("the time budget hands back", async () => {
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "open")) });
	try {
		const report = await runTask({ ...task, seconds: 0.3 }, r.deps, go());
		assert.equal(report.exit, "budget");
		assert.ok(report.steps.length >= 1);
	} finally {
		await r.close();
	}
});

test("an armd error ends the run with fault", async () => {
	const r = await rig();
	try {
		r.armd.fail("the serial bus is gone");
		const report = await runTask(task, r.deps, go());
		assert.equal(report.exit, "fault");
	} finally {
		await r.close();
	}
});

test("no wrist frame fails the tick; three in a row is a fault", async () => {
	const r = await rig({ cams: ["side"] });
	try {
		const report = await runTask(task, r.deps, go());
		assert.equal(report.exit, "fault");
		assert.match(report.reason, /3 ticks in a row failed/);
	} finally {
		await r.close();
	}
});

test("stopping from outside holds the arm and returns interrupted", async () => {
	const r = await rig();
	try {
		const stop = new AbortController();
		stop.abort();
		const report = await runTask(task, r.deps, stop.signal);
		assert.equal(report.exit, "interrupted");
		assert.equal(r.armd.calls.filter((c) => c === "POST /act").length, 0);
	} finally {
		await r.close();
	}
});

test("ask answers on the current frame without moving", async () => {
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "lower", { far: 1 })) });
	try {
		const asked = await askTask(task, r.deps);
		assert.equal(asked.reading.action, "lower");
		assert.deepEqual(asked.motion, { kind: "delta", height_m: -0.01 });
		assert.equal(r.armd.calls.filter((c) => c === "POST /act").length, 0);
	} finally {
		await r.close();
	}
});

test("an unsure Jev holds still and hands back after two ticks", async () => {
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "lower", { p: 0.2 })) });
	try {
		const report = await runTask(task, r.deps, go());
		assert.equal(report.exit, "unsure");
		assert.deepEqual(report.steps.map((s) => s.did), ["unsure; did nothing", "unsure; did nothing"]);
		assert.equal(r.armd.calls.filter((c) => c === "POST /act").length, 0);
	} finally {
		await r.close();
	}
});
