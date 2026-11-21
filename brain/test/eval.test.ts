import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import type { ArmState, Frame } from "../arm.ts";
import type { Answers } from "../decision.ts";
import { CHOICE_ID } from "../director.ts";
import { evaluate, offeredFor, phaseOf, requestFor, score, summarize, type Labelled, type Scored } from "../eval.ts";
import { recorded, Recorder } from "../record.ts";
import type { Facts, Skill } from "../skills.ts";
import { memorySink, scriptedDecider, tempDir } from "./helpers.ts";

const run = async () => ({ ok: true, why: "" });
const skills: Skill[] = [
	{ name: "find_object", says: "Look for it.", ready: (f) => !f.object_held, run },
	{ name: "nudge_left", says: "Move left.", amount: { says: "How far?", unit: "cm", levels: [1, 2] }, ready: (f) => f.object_located && f.jaw_tip_height_cm >= 11, run },
	{ name: "descend", says: "Go down.", ready: (f) => f.object_located, run },
	{ name: "close", says: "Close.", ready: (f) => f.jaw_tip_height_cm < 9, run },
];

const facts = (over: Partial<Facts> = {}): Facts => ({
	object_located: true,
	jaws_beside_object: true,
	jaw_tip_height_cm: 12,
	gripper: "open",
	object_held: false,
	over_set_down_spot: false,
	at_set_down_height: false,
	object_set_down: false,
	grip_failed_check: false,
	...over,
});

const arm = { pan_deg: 10, reach_m: 0.22, height_m: 0.12, pitch_deg: 55, roll_deg: -90, gripper_pct: 85, grip: "open" } as ArmState;

const item = (id: string, accepted: string[], f: Partial<Facts> = {}, ambiguous = false): Labelled => ({
	id,
	run: 1,
	tick: 1,
	frame: { sha256: "f".repeat(64), width: 640, height: 360 },
	state: { goal: "Pick up the bottle.", what: "bottle", held: null, arm, facts: facts(f), recent: [] },
	offered_then: [],
	jev_then: { choice: null, p: null },
	accepted,
	ambiguous,
	reason: "",
});

const answer = (probabilities: Record<string, number>): Answers => {
	const choice = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0]![0];
	return { [CHOICE_ID]: { type: "choice", choice, probabilities, confidence: probabilities[choice]! } };
};

test("offered skills come from today's preconditions on the recorded facts, worded as the prompts say", () => {
	const above = offeredFor(item("a", []), skills, { texts: {}, history: [] }).map((k) => k.name);
	assert.deepEqual(above, ["find_object", "nudge_left", "descend"]);
	const low = offeredFor(item("b", [], { jaw_tip_height_cm: 4 }), skills, { texts: { close: "Close when it sits between the jaws." }, history: [] });
	assert.deepEqual(low.map((k) => k.name), ["find_object", "descend", "close"]);
	assert.equal(low.find((k) => k.name === "close")!.says, "Close when it sits between the jaws.");
});

test("the request carries the variant's wording for the skills and the choice question", () => {
	const frame: Frame = { cam: "wrist", t_capture: 0, age_ms: 0, sha256: "f".repeat(64), width: 640, height: 360, mime: "image/jpeg", data_b64: "", upright: true };
	const { request } = requestFor(item("a", []), skills, { texts: { descend: "Down, when it is just right of the jaw.", choice: "Which skill?" }, history: [] }, frame, "on the wrist");
	const q = request.questions[CHOICE_ID]!;
	assert.equal(q.type, "choice");
	if (q.type !== "choice") return;
	assert.equal(q.instructions, "Which skill?");
	assert.equal(q.criteria.descend, "Down, when it is just right of the jaw.");
	assert.deepEqual(Object.keys(q.criteria), ["find_object", "nudge_left", "descend", "done", "stuck"]);
	assert.ok("amount_nudge_left" in request.questions);
});

test("a choice counts when it is accepted and on offer, and is acted on only when the director would run it", () => {
	const offered = ["find_object", "nudge_left", "descend"];
	const sure = score(item("a", ["descend", "close"]), offered, answer({ descend: 0.5, nudge_left: 0.3, find_object: 0.2 }));
	assert.deepEqual([sure.right, sure.acted, sure.wanted], [true, true, ["descend"]]);
	const close = score(item("b", ["descend"]), offered, answer({ descend: 0.3, nudge_left: 0.27, find_object: 0.43 - 0.3 }));
	assert.deepEqual([close.right, close.acted], [true, false], "0.03 ahead of the runner-up is unsure");
	const low = score(item("c", ["descend"]), offered, answer({ descend: 0.24, nudge_left: 0.1, find_object: 0.1, stuck: 0.56 - 0.2 }));
	assert.equal(low.choice, "stuck");
	assert.deepEqual([low.right, low.acted], [false, true]);
	const end = score(item("d", ["done"]), offered, answer({ done: 0.6, descend: 0.4 }));
	assert.deepEqual([end.right, end.acted], [true, true]);
	assert.equal(phaseOf(facts({ jaw_tip_height_cm: 4, gripper: "empty" })), "located, jaws low, jaws shut");
});

test("the summary counts acted-on picks and groups wrong ones by situation", () => {
	const row = (id: string, choice: string, right: boolean, acted: boolean, wanted = ["descend"]): Scored => ({ id, phase: "located, jaws above", ambiguous: false, offered: [], wanted, choice, p: 0.4, margin: 0.1, acted, right });
	const s = summarize([row("1", "descend", true, true), row("2", "nudge_left", false, true), row("3", "nudge_left", false, false), row("4", "close", false, true, ["raise"]), row("5", "descend", true, false)]);
	assert.deepEqual([s.n, s.right, s.acted, s.actedRight, s.actedWrong, s.unsure], [5, 2, 3, 1, 2, 2]);
	assert.equal(s.accuracy, 0.4);
	assert.equal(s.actedAccuracy, 0.333);
	assert.equal(s.unsureRate, 0.4);
	assert.deepEqual(s.confusions[0], { phase: "located, jaws above", wanted: "descend", picked: "nudge_left", n: 2, acted: 1, ids: ["2", "3"] });
	assert.equal(s.confusions.length, 2);
	assert.equal(summarize([], 2).actedAccuracy, 0);
});

test("evaluate asks once per decision, skips ones with nothing accepted on offer, and records each exchange", async () => {
	const dir = tempDir();
	writeFileSync(join(dir, `${"f".repeat(64)}.jpg`), Buffer.from([0xff, 0xd8, 0xff, 0xd9]));
	const sink = memorySink();
	const recorder = new Recorder(dir, sink);
	const asked: string[] = [];
	const model = recorded(
		scriptedDecider((req) => {
			asked.push(Object.keys((req.questions[CHOICE_ID] as { criteria: Record<string, string> }).criteria).join(","));
			return answer({ descend: 0.6, nudge_left: 0.2, find_object: 0.2 });
		}),
		recorder,
	);
	const items = [item("1.1", ["descend"]), item("1.2", ["close"]), item("1.3", ["close"], { jaw_tip_height_cm: 4 })];
	const { rows, skipped } = await evaluate(items, skills, { texts: {}, history: [] }, { model, recorder, view: "on the wrist" }, "t");
	assert.deepEqual(skipped, ["1.2"]);
	assert.deepEqual(rows.map((r) => [r.id, r.choice, r.right]), [["1.1", "descend", true], ["1.3", "descend", false]]);
	assert.equal(asked.length, 2);
	assert.equal(sink.records.filter((r) => r.kind === "valet.tick" && r.data.role === "replay").length, 2);
});
