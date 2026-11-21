import assert from "node:assert/strict";
import { test } from "node:test";
import { runProgram } from "../program.ts";
import { choose, rig, scriptedDecider } from "./helpers.ts";

const go = () => new AbortController().signal;

test("an error in the program ends it and is reported", async () => {
	const r = await rig();
	try {
		const out = await runProgram("p0", 'log("a"); throw new Error("boom");', r.deps, go(), 5);
		assert.equal(out.ok, false);
		assert.equal(out.error, "boom");
		assert.deepEqual(out.log, ["a"]);
	} finally {
		await r.close();
	}
});

test("a program that runs past its time is stopped and the arm holds", async () => {
	const r = await rig();
	try {
		const out = await runProgram("p0", "for (;;) await sleep(50);", r.deps, go(), 0.3);
		assert.equal(out.ok, false);
		assert.match(out.error ?? "", /ran past its 0.3 s/);
		assert.ok(r.armd.calls.includes("POST /hold"));
	} finally {
		await r.close();
	}
});

test("locate turns Jev's box into positions on a plane through armd", async () => {
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "stuck")) });
	try {
		const out = await runProgram("p0", 'log(await locate("bottle cap", { height_cm: 9 }));', r.deps, go(), 5);
		assert.equal(out.ok, true, out.error ?? "");
		const found = JSON.parse(out.log[0]!);
		assert.equal(found.found, true);
		assert.deepEqual(found.centre, { pan_deg: -13, reach_cm: 20, height_cm: 9 });
		assert.deepEqual(found.foot, { pan_deg: -12.6, reach_cm: 20, height_cm: 0 });
	} finally {
		await r.close();
	}
});

test("segment turns SAM 3's mask into positions on a plane through armd", async () => {
	const r = await rig();
	try {
		const code = 'log(await segment("blue bottle cap", { height_cm: 9 })); log(await segment("unicorn"));';
		const out = await runProgram("p0", code, r.deps, go(), 5);
		assert.equal(out.ok, true, out.error ?? "");
		const found = JSON.parse(out.log[0]!);
		assert.equal(found.found, true);
		assert.equal(found.whole, true);
		assert.deepEqual(found.centroid, { u: 0.5, v: 0.5 });
		assert.deepEqual(found.centre, { pan_deg: -13, reach_cm: 20, height_cm: 9 });
		assert.deepEqual(found.foot, { pan_deg: -12.6, reach_cm: 20, height_cm: 0 });
		assert.ok(found.ray && found.rays.bottom && /^[0-9a-f]{64}$/.test(found.frame));
		assert.deepEqual(JSON.parse(out.log[1]!), { found: false, score: 0.04, frame: found.frame });
		assert.deepEqual(r.armd.calls.filter((c) => c === "POST /segment").length, 2);
	} finally {
		await r.close();
	}
});

test("the program sees only the arm API", async () => {
	const r = await rig();
	try {
		const out = await runProgram("p0", 'log(typeof process, typeof require, typeof fetch, typeof moveTo);', r.deps, go(), 5);
		assert.deepEqual(out.log, ["undefined undefined undefined function"]);
	} finally {
		await r.close();
	}
});

test("triangulate takes two locate results and returns position and height", async () => {
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "stuck")) });
	try {
		const out = await runProgram("p0", 'const a = await locate("cap"); const b = await locate("cap"); log(await triangulate(a, b));', r.deps, go(), 5);
		assert.equal(out.ok, true, out.error ?? "");
		assert.deepEqual(JSON.parse(out.log[0]!), { ok: true, pan_deg: -14, reach_cm: 20, height_cm: 7, gap_cm: 0.4, angle_deg: 18 });
	} finally {
		await r.close();
	}
});

test("jawsAt centres the point in the open jaws, wider objects get half their width plus clearance", async () => {
	const { jawsAt, OPEN_GAP_CM } = await import("../program.ts");
	const deg = (cm: number, reach: number) => Math.round((20 - (cm / reach) * (180 / Math.PI)) * 10) / 10;
	assert.deepEqual(jawsAt({ pan_deg: 20, reach_cm: 20, height_cm: 4 }, { pitch_deg: 90, depth_cm: 2 }), { pan_deg: deg(OPEN_GAP_CM / 2, 20), reach_cm: 20, height_cm: 2, pitch_deg: 90 });
	assert.equal(jawsAt({ pan_deg: 20, reach_cm: 20, height_cm: 4 }, { pitch_deg: 90, width_cm: 9 }).pan_deg, deg(5, 20));
	const side = jawsAt({ pan_deg: 0, reach_cm: 25, height_cm: 4 }, { pitch_deg: 0 });
	assert.deepEqual([side.reach_cm, side.height_cm], [27, 4], "a level approach pushes the tips 2 cm past the point");
});

test("a move the arm cannot reach ends the program with the reason", async () => {
	const r = await rig({ maxReachM: 0.14 });
	try {
		const out = await runProgram("p0", 'log("before"); await move({ reach_cm: 5 }); log("after");', r.deps, go(), 5);
		assert.equal(out.ok, false);
		assert.match(out.error ?? "", /^refused: /);
		assert.deepEqual(out.log, ["before"]);
	} finally {
		await r.close();
	}
});
