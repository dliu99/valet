import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { after, before, test } from "node:test";
import type { Frame } from "../arm.ts";
import { readChatAnswers } from "../deciders/chat.ts";
import { systemOne } from "../deciders/systemone.ts";
import { DecisionError, type Exchange } from "../decision.ts";
import { ACTIONS, compile, MOVES, type Task } from "../task.ts";
import { jevAnswers, startSystemOneStub, type SystemOneStub } from "./systemone.ts";

const JPEG = readFileSync(new URL("./armd-double.jpg", import.meta.url));
const wrist: Frame = { cam: "wrist", t_capture: 0, age_ms: 5, sha256: "ab".repeat(32), width: 320, height: 240, mime: "image/jpeg", data_b64: JPEG.toString("base64"), upright: true };
const task: Task = { id: "0", goal: "Pick up the bottle.", seconds: 60 };
const arm = { t: 0, kind: "fake", allow_motion: true, joints: { shoulder_pan: 0, shoulder_lift: 0, elbow_flex: 0, wrist_flex: 0, wrist_roll: -90, gripper: 60 }, tip: { x: 0.2, y: 0, z: 0.1 }, pan_deg: 0, reach_m: 0.2, height_m: 0.1, pitch_deg: 60, roll_deg: -90, gripper_pct: 60, grip: "open", amps: {}, peak_amps: 0.1, torque: true } as const;
const request = () => compile(task, { arm: { ...arm, joints: { ...arm.joints }, tip: { ...arm.tip } }, frame: wrist, recent: [], view: "on the gripper" });

let server: SystemOneStub;
before(async () => {
	server = await startSystemOneStub();
});
after(() => server.close());

test("System One wire format: named image assets, a choice and a score per move", async () => {
	server.respond(200, jevAnswers());
	const model = systemOne({ id: "jev", baseUrl: server.baseUrl, model: "jev-latest", apiKey: "test-key" });
	let seen: Exchange | undefined;
	const answers = await model.decide(request(), { role: "ask", exchange: (e) => (seen = e) });

	const sent = server.requests.at(-1)!;
	assert.equal(sent.path, "/v1/systemone");
	assert.equal(sent.headers.authorization, "Bearer test-key");
	assert.equal(sent.headers["typesafe-danger-zone-images"], "true");
	const body = sent.body;
	assert.equal(body.model, "jev-latest");
	const ref = `wrist-${wrist.sha256.slice(0, 16)}`;
	assert.deepEqual(body.assets, { [ref]: { type: "image", format: "jpeg", data_base64: wrist.data_b64 } });
	assert.deepEqual(body.state.images, { wrist: ref }, "the image's reference sits in state where questions point at it");
	assert.equal(body.questions.far_reach_out.type, "score");
	assert.equal(body.questions.far_reach_out.criteria.length, 5);
	assert.deepEqual(Object.keys(body.questions.act.criteria), ACTIONS);

	assert.equal(answers.act?.type === "choice" && answers.act.choice, "reach_out");
	assert.deepEqual(answers.far_reach_out, { type: "score", score: 1.4, confidence: 0.5 });
	assert.deepEqual(seen?.usage, { input: 1290, output: 21 });
});

test("a local h2o server gets no key", async () => {
	server.respond(200, jevAnswers());
	await systemOne({ id: "h2o", baseUrl: server.baseUrl, model: "h2o-lightning-4b" }).decide(request(), { role: "ask" });
	assert.equal(server.requests.at(-1)!.headers.authorization, undefined);
});

test("a missing answer or an HTTP error rejects, after reporting the exchange", async () => {
	const partial = jevAnswers();
	delete (partial.answers as Record<string, unknown>).far_reach_out;
	server.respond(200, partial);
	const model = systemOne({ id: "jev", baseUrl: server.baseUrl, model: "jev-latest", apiKey: "k" });
	await assert.rejects(model.decide(request(), { role: "ask" }), (e: Error) => e instanceof DecisionError && /no answer to far_reach_out/.test(e.message));
	server.respond(503, { error: "overloaded" });
	let seen: Exchange | undefined;
	await assert.rejects(model.decide(request(), { role: "ask", exchange: (e) => (seen = e) }), /HTTP 503/);
	assert.deepEqual(seen?.response, { status: 503, body: { error: "overloaded" } });
});

test("the chat decider's JSON reads into the same Answer shape", () => {
	const questions = request().questions;
	const reply = { act: { probabilities: { reach_out: 0.75, lower: 0.25 } }, ...Object.fromEntries(Object.keys(MOVES).map((m) => [`far_${m}`, { score: 2.5, confidence: 0.6 }])) };
	const answers = readChatAnswers(questions, `Here: ${JSON.stringify(reply)}`);
	assert.equal(answers.act?.type === "choice" && answers.act.choice, "reach_out");
	assert.equal(answers.act?.type === "choice" && answers.act.probabilities.reach_out, 0.75);
	assert.deepEqual(answers.far_reach_out, { type: "score", score: 2.5, confidence: 0.6 }, "a score keeps its level position, not a 0-1 clamp");
	assert.throws(() => readChatAnswers(questions, '{"act": {}}'), /no answer to far_pan_left/);
});
