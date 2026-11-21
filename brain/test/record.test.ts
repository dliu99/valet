import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { after, before, test } from "node:test";
import type { Frame } from "../arm.ts";
import { systemOne } from "../deciders/systemone.ts";
import { conversationSink, ctx, openStore, recorded, recordedTicks, Recorder, RunDoc, RunIndex, startRun, TickEntry, type TickData } from "../record.ts";
import { reask } from "../stage.ts";
import { compile, type Task } from "../task.ts";
import { memorySink, tempDir } from "./helpers.ts";
import { jevAnswers, startSystemOneStub, type SystemOneStub } from "./systemone.ts";

const JPEG = readFileSync(new URL("./armd-double.jpg", import.meta.url));
const SHA = createHash("sha256").update(JPEG).digest("hex");
const wrist: Frame = { cam: "wrist", t_capture: 0, age_ms: 5, sha256: SHA, width: 320, height: 240, mime: "image/jpeg", data_b64: JPEG.toString("base64"), upright: true };
const task: Task = { id: "0", goal: "Pick up the bottle.", seconds: 60 };
const arm = { t: 0, kind: "fake" as const, allow_motion: true, joints: { shoulder_pan: 0, shoulder_lift: 0, elbow_flex: 0, wrist_flex: 0, wrist_roll: -90, gripper: 60 }, tip: { x: 0.2, y: 0, z: 0.1 }, pan_deg: 0, reach_m: 0.2, height_m: 0.1, pitch_deg: 60, roll_deg: -90, gripper_pct: 60, grip: "open" as const, amps: {}, peak_amps: 0.1, torque: true };
const request = () => compile(task, { arm, frame: wrist, recent: [], view: "on the gripper" });
const reading = { action: "reach_out" as const, p: 0.8 };

let jev: SystemOneStub;
before(async () => {
	jev = await startSystemOneStub();
});
after(() => jev.close());

test("a decision is recorded before its answers return, with image bytes replaced by the frame's hash", async () => {
	jev.respond(200, jevAnswers());
	const sink = memorySink();
	const framesDir = join(tempDir(), "frames");
	const model = recorded(systemOne({ id: "jev", baseUrl: jev.baseUrl, model: "jev-latest", apiKey: "k" }), new Recorder(framesDir, sink));
	const answers = await model.decide(request(), {
		role: "tick",
		task: task.id,
		tick: 7,
		facts: { arm, recent: [], framesMs: 31 },
		plan: () => ({ reading, motion: { kind: "delta", reach_m: 0.04 }, step: "4.0 cm" }),
	});
	assert.ok(answers.act);
	assert.equal(sink.records.length, 1, "recorded by the time the answers come back");

	const tick = sink.records[0]!;
	assert.equal(tick.kind, TickEntry.kind);
	const data = tick.data as TickData;
	assert.deepEqual([data.task, data.tick, data.role, data.decider, data.error], [task.id, 7, "tick", "jev", null]);
	assert.deepEqual(data.frames, [{ cam: "wrist", sha256: SHA, width: 320, height: 240 }]);
	assert.deepEqual(readFileSync(join(framesDir, `${SHA}.jpg`)), JPEG, "the bytes Jev saw, under their hash");
	const sent = (data.wire.request as any).assets[`wrist-${SHA.slice(0, 16)}`];
	assert.equal(sent.data_base64, SHA, "image data in the wire payload is the frame's hash");
	assert.ok(!JSON.stringify(data).includes(wrist.data_b64.slice(0, 200)), "no image bytes anywhere in the record");
	assert.deepEqual(data.reading, reading);
	assert.deepEqual(data.motion, { kind: "delta", reach_m: 0.04 });
	assert.deepEqual(data.usage, { input: 1290, output: 21 });
	assert.equal(data.ms.frames, 31);
	assert.equal((data.answers as any).far_reach_out.score, 1.4);
});

test("a failed call is recorded too, with what came back", async () => {
	jev.respond(500, { error: "boom" });
	const sink = memorySink();
	const model = recorded(systemOne({ id: "jev", baseUrl: jev.baseUrl, model: "jev-latest", apiKey: "k" }), new Recorder(join(tempDir(), "frames"), sink));
	await assert.rejects(model.decide(request(), { role: "tick", task: task.id, tick: 1, plan: () => ({ reading, motion: null, step: null }) }), /HTTP 500/);
	assert.equal(sink.records.length, 1);
	const data = sink.records[0]!.data as TickData;
	assert.match(data.error ?? "", /HTTP 500/);
	assert.equal(data.answers, null);
	assert.equal(data.reading, null);
	assert.deepEqual(data.wire.response, { status: 500, body: { error: "boom" } });
	assert.equal((data.wire.request as any).model, "jev-latest");
});

test("planner requests keep their images as frame hashes", async () => {
	const sink = memorySink();
	const framesDir = join(tempDir(), "frames");
	const recorder = new Recorder(framesDir, sink);
	const b64 = JPEG.toString("base64");
	await recorder.llmRequest(
		{ provider: "anthropic", id: "claude-opus-5-5", api: "anthropic-messages" },
		{ messages: [{ role: "user", content: [{ type: "text", text: "look" }, { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } }, { type: "image_url", url: `data:image/jpeg;base64,${b64}` }] }] },
	);
	const data = sink.records[0]!.data as any;
	assert.equal(sink.records[0]!.kind, "valet.llm_request");
	assert.equal(data.model, "anthropic/claude-opus-5-5");
	const content = data.body.messages[0].content;
	assert.equal(content[0].text, "look");
	assert.equal(content[1].source.data, SHA);
	assert.equal(content[2].url, SHA);
	assert.ok(existsSync(join(framesDir, `${SHA}.jpg`)));
});

test("records land in the run's conversation in the store, and a recorded tick can be asked again", async () => {
	jev.respond(200, jevAnswers());
	const dir = tempDir();
	const store = await openStore(dir);
	try {
		const id = await startRun(store, { kind: "run", goal: "move the white box", planner: "scripted/pick-and-move", decider: "jev" });
		const recorder = new Recorder(store.framesDir, conversationSink(store, id));
		const model = recorded(systemOne({ id: "jev", baseUrl: jev.baseUrl, model: "jev-latest", apiKey: "k" }), recorder);
		await model.decide(request(), { role: "tick", task: task.id, tick: 1, facts: { arm, recent: [] }, plan: () => ({ reading, motion: null, step: null }) });

		const root = await store.harness.root(ctx);
		const index = await store.harness.snapshot(RunIndex, root.id, ctx);
		assert.deepEqual(index?.runs.map((r) => [r.conversationId, r.kind, r.goal]), [[id, "run", "move the white box"]]);
		assert.equal((await store.harness.snapshot(RunDoc, id, ctx))?.status, "running");
		const ticks = await recordedTicks(store, id);
		assert.equal(ticks.length, 1);
		assert.equal(ticks[0]!.frames[0]!.sha256, SHA);

		jev.respond(200, jevAnswers("lower"));
		const edited: Task = { ...task, goal: "Lower the open gripper around the bottle's neck." };
		const again = await reask(edited, ticks[0]!, { model, recorder, view: "on the gripper" });
		assert.equal(again.reading.action, "lower");
		const sent = jev.requests.at(-1)!.body;
		assert.equal(sent.state.goal, edited.goal, "the edited goal was asked");
		assert.equal(sent.assets[`wrist-${SHA.slice(0, 16)}`].data_base64, wrist.data_b64, "on the recorded frame's bytes");
		assert.equal((await recordedTicks(store, id)).length, 1, "a re-ask is recorded with role replay, not as a tick");
	} finally {
		await store.close();
	}
});
