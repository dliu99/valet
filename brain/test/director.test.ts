import assert from "node:assert/strict";
import { test } from "node:test";
import type { DecisionRequest } from "../decision.ts";
import { runGoal, type Goal } from "../director.ts";
import type { Look, Overseer } from "../overseer.ts";
import type { Scratch, Skill, SkillArgs, SkillResult } from "../skills.ts";
import { choose, rig, scriptedDecider } from "./helpers.ts";

const go = () => new AbortController().signal;
const goal: Goal = { text: "Move the bottle a little to the left.", what: "bottle" };

function skills(over: Partial<Record<string, Skill["run"]>> = {}) {
	const ran: { name: string; args: SkillArgs }[] = [];
	const make = (name: string, says: string, amount?: Skill["amount"]): Skill => ({
		name,
		says,
		...(amount ? { amount } : {}),
		run: async (r, args, scratch): Promise<SkillResult> => {
			ran.push({ name, args });
			return over[name] ? over[name]!(r, args, scratch) : { ok: true, why: `${name} worked` };
		},
	});
	const list = [
		make("grab", "Grab the object and lift it."),
		make("raise", "Raise the gripper straight up.", { says: "How far up should it go?", unit: "cm", levels: [1, 2, 4] }),
	];
	return { list, ran };
}

test("Jev's choice runs that skill with the scored amount, and the next request sees the result", async () => {
	const { list, ran } = skills({ raise: async (r, a) => ({ ok: true, why: `raised: ${(await r.move({ height_cm: a.amount })).result}` }) });
	const seen: DecisionRequest[] = [];
	const r = await rig({ decider: () => scriptedDecider((req, n) => (seen.push(req), choose(req, n === 1 ? "raise" : "done", { far: 1.5 }))) });
	try {
		const report = await runGoal(goal, r.deps, go(), list);
		assert.equal(report.exit, "done");
		assert.deepEqual(ran, [{ name: "raise", args: { what: "bottle", amount: 3 } }]);
		assert.deepEqual(
			r.sink.records.filter((x) => x.kind === "valet.motion").map((m) => m.data.motion),
			[{ kind: "delta", height_m: 0.03 }],
		);
		assert.deepEqual(report.steps.map(({ ms: _, ...s }) => s), [
			{ skill: "raise", amount: 3, p: 0.9, ok: true, why: "raised: moved" },
			{ skill: "done", p: 0.9, ok: true, why: "Jev judged the goal accomplished" },
		]);

		const first = seen[0]!;
		const choice = first.questions.skill!;
		assert.equal(choice.type, "choice");
		assert.deepEqual(choice.type === "choice" && choice.criteria, {
			grab: "Grab the object and lift it.",
			raise: "Raise the gripper straight up.",
			done: "`goal` is accomplished.",
			stuck: "`goal` cannot be accomplished with these skills from here.",
		});
		assert.deepEqual(Object.keys(first.questions), ["skill", "amount_raise"], "an amount question only for a skill with an amount");
		const far = first.questions.amount_raise!;
		assert.equal(far.instructions, 'If the next skill is "Raise the gripper straight up.": How far up should it go?');
		assert.deepEqual(far.type === "score" && far.criteria, ["About 1 cm.", "About 2 cm.", "About 4 cm."]);
		const state = first.state as Record<string, any>;
		assert.deepEqual([state.goal, state.object, state.held, state.recent_steps], [goal.text, "bottle", "nothing", []]);
		assert.equal(state.gripper, "unknown");
		assert.deepEqual(Object.keys(first.images), ["wrist"]);
		assert.deepEqual((seen[1]!.state as Record<string, any>).recent_steps, [{ skill: "raise", amount: "3 cm", ok: true, result: "raised: moved" }]);
	} finally {
		await r.close();
	}
});

test("what a skill holds is in the next request", async () => {
	const { list } = skills({ grab: async (_r, a, scratch) => ((scratch.held = { what: a.what, grasp: { pan_deg: 0, reach_cm: 20, height_cm: 3 } }), { ok: true, why: "holding" }) });
	const seen: DecisionRequest[] = [];
	const r = await rig({ decider: () => scriptedDecider((req, n) => (seen.push(req), choose(req, n === 1 ? "grab" : "stuck"))) });
	try {
		const report = await runGoal(goal, r.deps, go(), list);
		assert.equal(report.exit, "stuck");
		assert.equal((seen[1]!.state as Record<string, any>).held, "bottle");
	} finally {
		await r.close();
	}
});

test("done on the first look ends the goal without running a skill", async () => {
	const { list, ran } = skills();
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "done")) });
	try {
		const report = await runGoal(goal, r.deps, go(), list);
		assert.equal(report.exit, "done");
		assert.deepEqual(ran, []);
		assert.equal(r.armd.calls.filter((c) => c === "POST /act").length, 0);
	} finally {
		await r.close();
	}
});

test("an unsure Jev runs nothing and hands back after two steps in a row", async () => {
	const { list, ran } = skills();
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "grab", { p: 0.2 })) });
	try {
		const report = await runGoal(goal, r.deps, go(), list);
		assert.equal(report.exit, "unsure");
		assert.deepEqual(ran, []);
		assert.deepEqual(report.steps.map((s) => [s.skill, s.ok, s.why]), [["grab", false, "unsure; ran nothing"], ["grab", false, "unsure; ran nothing"]]);
	} finally {
		await r.close();
	}
});

test("a skill that throws is a failed step Jev sees, and the goal goes on", async () => {
	const { list, ran } = skills({ grab: async () => { throw new Error("the jaws jammed"); } });
	const seen: DecisionRequest[] = [];
	const r = await rig({ decider: () => scriptedDecider((req, n) => (seen.push(req), choose(req, n! < 3 ? "grab" : "stuck"))) });
	try {
		const report = await runGoal(goal, r.deps, go(), list);
		assert.equal(report.exit, "stuck");
		assert.equal(ran.length, 2);
		assert.deepEqual(report.steps.map((s) => [s.skill, s.ok, s.why]), [["grab", false, "the jaws jammed"], ["grab", false, "the jaws jammed"], ["stuck", false, "Jev judged the goal out of reach of these skills"]]);
		assert.deepEqual((seen[2]!.state as Record<string, any>).recent_steps, [
			{ skill: "grab", ok: false, result: "the jaws jammed" },
			{ skill: "grab", ok: false, result: "the jaws jammed" },
		]);
	} finally {
		await r.close();
	}
});

test("stopping from outside ends the goal: the running skill's next move throws and the arm holds", async () => {
	const stop = new AbortController();
	const { list, ran } = skills({ raise: async (r) => (stop.abort(), await r.move({ height_cm: 1 }), { ok: true, why: "raised" }) });
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "raise")) });
	try {
		const report = await runGoal(goal, r.deps, stop.signal, list);
		assert.equal(report.exit, "interrupted");
		assert.equal(ran.length, 1);
		assert.equal(report.steps[0]!.ok, false);
		assert.equal(r.armd.calls.filter((c) => c === "POST /act").length, 0);
		assert.equal(r.armd.calls.filter((c) => c === "POST /hold").length, 2, "held at the start and again on the stop");
	} finally {
		await r.close();
	}
});

test("a goal stopped before it starts runs nothing", async () => {
	const { list, ran } = skills();
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "grab")) });
	try {
		const stop = new AbortController();
		stop.abort();
		const report = await runGoal(goal, r.deps, stop.signal, list);
		assert.equal(report.exit, "interrupted");
		assert.deepEqual([ran, report.steps], [[], []]);
	} finally {
		await r.close();
	}
});

test("the step budget hands back", async () => {
	const { list, ran } = skills();
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "grab")) });
	try {
		const report = await runGoal({ ...goal, steps: 3 }, r.deps, go(), list);
		assert.equal(report.exit, "budget");
		assert.equal(ran.length, 3);
	} finally {
		await r.close();
	}
});

test("no wrist frame fails the goal", async () => {
	const { list, ran } = skills();
	const r = await rig({ cams: ["side"], decider: () => scriptedDecider((req) => choose(req, "grab")) });
	try {
		const report = await runGoal(goal, r.deps, go(), list);
		assert.equal(report.exit, "failed");
		assert.match(report.reason, /no wrist frame/);
		assert.deepEqual(ran, []);
	} finally {
		await r.close();
	}
});

test("each step is recorded: the decision as sent and answered, and what the skill did", async () => {
	const { list } = skills({ raise: async (r) => (r.log("up we go"), { ok: true, why: "raised", height_cm: 12 }) });
	const r = await rig({ decider: () => scriptedDecider((req, n) => choose(req, n === 1 ? "raise" : "done", { far: 2 })) });
	try {
		await runGoal({ ...goal, id: "g1" }, r.deps, go(), list);
		const decisions = r.sink.records.filter((x) => x.kind === "valet.tick").map((x) => x.data as Record<string, any>);
		assert.deepEqual(decisions.map((d) => [d.role, d.task, d.tick, d.reading, d.step]), [
			["step", "g1", 1, { action: "raise", p: 0.9 }, "4 cm"],
			["step", "g1", 2, { action: "done", p: 0.9 }, null],
		]);
		assert.ok(decisions[0]!.wire.request && decisions[0]!.answers.skill, "the wire request and the answers are kept");
		assert.equal(decisions[0]!.state.goal, goal.text);
		const steps = r.sink.records.filter((x) => x.kind === "valet.step").map((x) => x.data as Record<string, any>);
		assert.deepEqual(steps.map(({ ms: _, ...s }) => s), [
			{ task: "g1", step: 1, goal: goal.text, what: "bottle", skill: "raise", amount: 4, unit: "cm", p: 0.9, ok: true, why: "raised", result: { ok: true, why: "raised", height_cm: 12 }, log: ["up we go"], held: null },
			{ task: "g1", step: 2, goal: goal.text, what: "bottle", skill: "done", amount: null, unit: null, p: 0.9, ok: true, why: "Jev judged the goal accomplished", result: null, log: [], held: null },
		]);
	} finally {
		await r.close();
	}
});

test("a skill may not be called done or stuck", async () => {
	const r = await rig();
	try {
		await assert.rejects(runGoal(goal, r.deps, go(), [{ name: "done", says: "x", run: async () => ({ ok: true, why: "" }) }]), /done and stuck are reserved/);
	} finally {
		await r.close();
	}
});

test("a skill that fails the same way three times in a row ends the goal as stuck", async () => {
	const { list, ran } = skills({ grab: async () => ({ ok: false, why: "refused: out of reach" }) });
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "grab")) });
	try {
		const report = await runGoal(goal, r.deps, go(), list);
		assert.equal(report.exit, "stuck");
		assert.equal(ran.length, 3);
		assert.match(report.reason, /grab failed the same way 3 times/);
	} finally {
		await r.close();
	}
});

test("only skills whose preconditions hold are offered, and a lone one runs without asking", async () => {
	const { list, ran } = skills();
	const gated = list.map((k) => (k.name === "raise" ? { ...k, ready: () => false } : k));
	const seen: DecisionRequest[] = [];
	const r = await rig({ decider: () => scriptedDecider((req) => (seen.push(req), choose(req, "done"))) });
	try {
		const report = await runGoal({ ...goal, steps: 2 }, r.deps, go(), gated);
		assert.deepEqual(ran.map((x) => x.name), ["grab", "grab"]);
		assert.equal(seen.length, 0);
		assert.equal(report.exit, "budget");
		assert.match(report.steps[0]!.why, /Jev was not asked/);
	} finally {
		await r.close();
	}
});

function overseerSaying(answers: Partial<Record<string, boolean>>) {
	const looks: Look[] = [];
	const overseer: Overseer = {
		async look(l) {
			looks.push(l);
			const agree = answers[l.kind] ?? true;
			return { agree, why: agree ? "looks right" : "the pictures show otherwise", edited: [], refused: [] };
		},
	};
	return { overseer, looks };
}

test("Claude is asked after a skill with a checkpoint succeeds, and a disagreement undoes its claim", async () => {
	const { list, ran } = skills({ grab: async (_r, _a, s) => ((s.target = { pan_deg: 0, reach_cm: 20, height_cm: 0 }), { ok: true, why: "found it" }) });
	const checked = list.map((k) => (k.name === "grab" ? { ...k, checkpoint: { kind: "found" as const, refute: (s: Scratch) => void (s.target = undefined) } } : k));
	const { overseer, looks } = overseerSaying({ found: false });
	const seen: DecisionRequest[] = [];
	const scratch: Scratch = {};
	const r = await rig({ decider: () => scriptedDecider((req, n) => (seen.push(req), choose(req, n === 1 ? "grab" : "stuck"))) });
	try {
		const report = await runGoal(goal, r.deps, go(), checked, scratch, { overseer });
		assert.deepEqual(ran.map((x) => x.name), ["grab"]);
		assert.deepEqual(looks.map((l) => l.kind), ["found", "handback"]);
		assert.ok(looks[0]!.frames.length >= 1, "Claude sees the pictures");
		assert.equal(scratch.target, undefined, "the claim is undone");
		assert.equal(report.exit, "stuck");
		const recent = (seen[1]!.state as Record<string, any>).recent_steps;
		assert.match(recent.at(-1).result, /^Claude disagrees/);
	} finally {
		await r.close();
	}
});

test("done needs Claude's agreement; without it Jev keeps going", async () => {
	const { list } = skills();
	let doneAsked = 0;
	const overseer: Overseer = { look: async (l) => ({ agree: l.kind === "done" && ++doneAsked > 1, why: "w", edited: [], refused: [] }) };
	const r = await rig({ decider: () => scriptedDecider((req) => choose(req, "done")) });
	try {
		const report = await runGoal({ ...goal, steps: 5 }, r.deps, go(), list, {}, { overseer });
		assert.equal(report.exit, "done");
		assert.equal(doneAsked, 2);
		assert.match(report.reason, /Claude agreed/);
	} finally {
		await r.close();
	}
});

test("Jev reads Claude's rewrites of the skills and of the question", async () => {
	const { list } = skills();
	const seen: DecisionRequest[] = [];
	const prompts = () => ({ texts: { choice: "Pick the next skill.", grab: "Close on the bottle once it sits between the jaws." }, history: [] });
	const r = await rig({ decider: () => scriptedDecider((req) => (seen.push(req), choose(req, "stuck"))) });
	try {
		await runGoal(goal, r.deps, go(), list, {}, { prompts });
		const q = seen[0]!.questions.skill as { instructions: string; criteria: Record<string, string> };
		assert.equal(q.instructions, "Pick the next skill.");
		assert.equal(q.criteria.grab, "Close on the bottle once it sits between the jaws.");
	} finally {
		await r.close();
	}
});

test("at a handback Claude may name the next step; code runs it and the goal goes on", async () => {
	const { list, ran } = skills();
	let handbacks = 0;
	const overseer: Overseer = {
		look: async (l) => (l.kind === "handback" && ++handbacks === 1 ? { agree: true, why: "grab it", next: { skill: "grab" }, edited: [], refused: [] } : { agree: true, why: "ok", edited: [], refused: [] }),
	};
	const r = await rig({ decider: () => scriptedDecider((req, n) => choose(req, n! <= 2 ? "grab" : "done", { p: n! <= 2 ? 0.1 : 0.9 })) });
	try {
		const report = await runGoal({ ...goal, steps: 6 }, r.deps, go(), list, {}, { overseer });
		assert.deepEqual(ran.map((x) => x.name), ["grab"]);
		assert.equal(handbacks, 1);
		assert.equal(report.exit, "done");
		assert.match(report.steps.find((s) => s.skill === "grab" && s.ok)!.why, /Claude's choice/);
	} finally {
		await r.close();
	}
});
