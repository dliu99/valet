import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { ArmState, Frame } from "../arm.ts";
import type { Answers } from "../decision.ts";
import { ACTIONS, compile, MOVES, read, toMotion, validate, type Action, type Step, type Task } from "../task.ts";

function arm(patch: Partial<ArmState> = {}): ArmState {
	return {
		t: 0,
		kind: "fake",
		allow_motion: true,
		joints: { shoulder_pan: 0, shoulder_lift: 15, elbow_flex: 20, wrist_flex: 45, wrist_roll: -90, gripper: 60 },
		tip: { x: 0.25, y: 0, z: 0.1 },
		pan_deg: 0,
		reach_m: 0.25,
		height_m: 0.1,
		pitch_deg: 30,
		roll_deg: -90,
		gripper_pct: 60,
		grip: "open",
		amps: {},
		peak_amps: 0.1,
		torque: true,
		...patch,
	};
}

const frame: Frame = { cam: "wrist", t_capture: 0, age_ms: 5, sha256: "a".repeat(64), width: 640, height: 480, mime: "image/jpeg", data_b64: "AAAA", upright: true };
const task: Task = { id: "0", goal: "Pick up the bottle.", seconds: 60 };
const act = (choice: Action, p = 0.9): Answers["act"] => ({ type: "choice", choice, probabilities: Object.fromEntries(ACTIONS.map((a) => [a, a === choice ? p : (1 - p) / (ACTIONS.length - 1)])), confidence: p });
const far = (score: number): Answers["x"] => ({ type: "score", score, confidence: 0.6 });

describe("compile", () => {
	test("one question for the action over the whole vocabulary, and a distance per move", () => {
		const recent: Step[] = [{ tick: 1, action: "lower", p: 0.8, step: "2.0 cm", did: "refused: unreachable" }];
		const req = compile({ ...task, notes: "The bottle is left of the tape." }, { arm: arm({ grip: "holding" }), frame, recent, view: "on the gripper" });
		assert.deepEqual(Object.keys(req.questions), ["act", ...Object.keys(MOVES).map((m) => `far_${m}`)]);
		assert.deepEqual(Object.keys(req.questions.act!.type === "choice" ? req.questions.act!.criteria : {}), ACTIONS);
		assert.match(req.questions.far_reach_out!.instructions, /^If the next action is "Move the gripper straight away from the base/);
		assert.equal(req.questions.far_reach_out!.type === "score" && req.questions.far_reach_out!.criteria.length, 5);
		const state = req.state as Record<string, any>;
		assert.equal(state.goal, "Pick up the bottle.");
		assert.equal(state.notes, "The bottle is left of the tape.");
		assert.match(state.gripper, /^holding/);
		assert.deepEqual(state.recent_actions, [{ action: "lower", step: "2.0 cm", result: "refused: unreachable" }]);
		assert.match(state.camera, /image up is away from the robot's base/);
		assert.deepEqual(Object.keys(req.images), ["wrist"]);
	});

	test("a task needs a goal and a time budget", () => {
		assert.deepEqual(validate(task), []);
		assert.match(validate({ ...task, goal: " ", seconds: 0 }).join("\n"), /goal:[\s\S]*seconds:/);
	});
});

describe("action to motion", () => {
	test("Jev's distance answer sets the step, between its levels", () => {
		const cases: [Action, number, Record<string, number>, string][] = [
			["reach_out", 2, { reach_m: 0.02 }, "2.0 cm"],
			["reach_in", 0, { reach_m: -0.005 }, "0.5 cm"],
			["raise", 2.5, { height_m: 0.03 }, "3.0 cm"],
			["lower", 4, { height_m: -0.08 }, "8.0 cm"],
			["pitch_down", 1, { pitch_deg: 5 }, "5.0 °"],
			["roll_ccw", 4, { roll_deg: -90 }, "90.0 °"],
		];
		for (const [action, score, delta, step] of cases)
			assert.deepEqual(toMotion(action, { act: act(action), [`far_${action}`]: far(score) }, arm()), { motion: { kind: "delta", ...delta }, step }, action);
	});

	test("a pan travels the jaw tip the judged distance at the current reach", () => {
		const turn = (reach: number) => (toMotion("pan_right", { act: act("pan_right"), far_pan_right: far(2) }, arm({ reach_m: reach })).motion as { pan_deg: number }).pan_deg;
		assert.ok(Math.abs(turn(0.25) - (0.02 / 0.25) * (180 / Math.PI)) < 0.01);
		assert.ok(turn(0.125) > turn(0.25), "nearer the base, the same travel is a bigger turn");
	});

	test("open and close drive the gripper; done and stuck move nothing", () => {
		assert.deepEqual(toMotion("open", { act: act("open") }, arm()).motion, { kind: "grip", to: "open" });
		assert.deepEqual(toMotion("close", { act: act("close") }, arm()).motion, { kind: "grip", to: "close" });
		for (const a of ["done", "stuck"] as Action[]) assert.equal(toMotion(a, { act: act(a) }, arm()).motion, null);
	});

	test("the top choice is the action, whatever its probability", () => {
		assert.deepEqual(read({ act: act("lower", 0.31) }), { action: "lower", p: 0.31 });
	});
});
