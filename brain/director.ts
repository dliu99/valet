import type { ArmState, Frame } from "./arm.ts";
import { topChoice, type Answers, type DecisionRequest, type JsonObject, type Question } from "./decision.ts";
import { makeRobot, settle, type ProgramDeps } from "./program.ts";
import type { DecisionPlan } from "./record.ts";
import type { CheckpointKind, Overseer } from "./overseer.ts";
import { applyPrompts, currentTexts, DEFAULT_CHOICE, type Prompts } from "./prompts.ts";
import { factsOf, SKILLS, type Facts, type Scratch, type Skill, type SkillArgs, type SkillResult } from "./skills.ts";
import { armFacts, levelValue, RECENT, UNSURE_HANDBACK, WRIST_CONVENTION, type Level } from "./task.ts";

/** A goal for Jev to reach by choosing skills. `what` is the object the skills act on. */
export interface Goal {
	text: string;
	what: string;
	/** At most this many decisions (default DEFAULT_STEPS). */
	steps?: number;
	/** Names this goal's records; default "do". */
	id?: string;
}

export type Exit = "done" | "stuck" | "unsure" | "failed" | "budget" | "interrupted";

export interface GoalStep {
	/** The skill run, or done, stuck, or (when unsure) the choice Jev leaned to. */
	skill: string;
	amount?: number;
	p: number;
	ok: boolean;
	why: string;
	ms: number;
}

export interface GoalReport {
	exit: Exit;
	reason: string;
	steps: GoalStep[];
}

export const DEFAULT_STEPS = 24;
/** With about ten skills on offer, a top choice this likely, and this far ahead of the next, is acted on. */
export const SKILL_SURE = 0.25;
export const SKILL_MARGIN = 0.05;
/** A skill that fails the same way this many times in a row ends the goal: choosing it again will not help. */
export const REPEATS = 3;
/** How many times a goal may go on at Claude's say-so after Jev stops. */
export const MAX_REPLANS = 3;
export const CHOICE_ID = "skill";
export const amountId = (skill: string) => `amount_${skill}`;

const ENDS = {
	done: "`goal` is accomplished.",
	stuck: "`goal` cannot be accomplished with these skills from here.",
} as const;
type End = keyof typeof ENDS;
const isEnd = (choice: string): choice is End => choice in ENDS;

const UNIT_WORD = { cm: "cm", deg: "degrees" } as const;
const levels = (skill: Skill): Level[] => (skill.amount?.levels ?? []).map((value) => ({ value, label: `About ${value} ${UNIT_WORD[skill.amount!.unit]}.` }));
const amountText = (skill: Skill | undefined, amount: number | undefined) => (skill?.amount && amount !== undefined ? `${amount} ${skill.amount.unit === "cm" ? "cm" : "°"}` : null);

export interface Situation {
	arm: ArmState;
	frame: Frame;
	held: Scratch["held"];
	facts: Facts;
	recent: readonly GoalStep[];
	view: string;
}

/** The skills whose preconditions hold on these facts: the only ones the chooser is offered. */
export const readySkills = (skills: readonly Skill[], f: Facts) => skills.filter((k) => !k.ready || k.ready(f));

/** One request to Jev: which skill next toward the goal (or done, or stuck), and for each skill with an amount, how much. */
export function compileGoal(goal: Goal, skills: readonly Skill[], s: Situation, choice = DEFAULT_CHOICE): DecisionRequest {
	const state: JsonObject = {
		goal: goal.text,
		object: goal.what,
		held: s.held ? s.held.what : "nothing",
		...armFacts(s.arm),
		facts: { ...s.facts },
		recent_steps: s.recent.slice(-RECENT).map((step): JsonObject => {
			const amount = amountText(skills.find((k) => k.name === step.skill), step.amount);
			return { skill: step.skill, ...(amount ? { amount } : {}), ok: step.ok, result: step.why };
		}),
		camera: `${s.view}. ${WRIST_CONVENTION}`,
	};
	const questions: Record<string, Question> = {
		[CHOICE_ID]: {
			type: "choice",
			instructions: choice,
			criteria: { ...Object.fromEntries(skills.map((k) => [k.name, k.says])), ...ENDS },
		},
	};
	for (const k of skills)
		if (k.amount) questions[amountId(k.name)] = { type: "score", instructions: `If the next skill is "${k.says}": ${k.amount.says}`, criteria: levels(k).map((l) => l.label) };
	return { state, images: { wrist: s.frame }, questions };
}

export interface GoalReading {
	choice: string;
	p: number;
	/** How far the top choice is ahead of the next one. */
	margin: number;
	skill?: Skill;
	amount?: number;
}

/** The top choice, and the amount Jev scored for it, between the skill's levels. */
export function readGoal(answers: Answers, skills: readonly Skill[]): GoalReading {
	const choice = answers[CHOICE_ID];
	const ranked = choice?.type === "choice" ? Object.values(choice.probabilities ?? {}).sort((x, y) => y - x) : [];
	const pick = topChoice(choice);
	const top = pick && { ...pick, margin: ranked.length > 1 ? ranked[0]! - ranked[1]! : 1 };
	if (top && isEnd(top.choice)) return top;
	const skill = top && skills.find((k) => k.name === top.choice);
	if (!top || !skill) throw new Error(`the ${CHOICE_ID} answer named no skill: ${JSON.stringify(answers[CHOICE_ID])}`);
	if (!skill.amount) return { ...top, skill };
	const a = answers[amountId(skill.name)];
	const amount = Math.round(levelValue(levels(skill), a?.type === "score" ? a.score : 0) * 10) / 10;
	return { ...top, skill, amount };
}

export interface SkillRun {
	result: SkillResult;
	log: string[];
	ms: number;
}

/** Runs one skill on a fresh robot; a skill that throws is a failed result with the error as `why`. */
export async function runSkill(skill: Skill, args: SkillArgs, scratch: Scratch, deps: ProgramDeps, signal: AbortSignal, id: string): Promise<SkillRun> {
	const log: string[] = [];
	const t0 = performance.now();
	let result: SkillResult;
	try {
		result = await skill.run(makeRobot(id, deps, signal, log), args, scratch);
	} catch (e) {
		result = { ok: false, why: (e as Error)?.message ?? String(e) };
	}
	scratch.last = { skill: skill.name, ok: result.ok, why: result.why };
	return { result, log, ms: Math.round(performance.now() - t0) };
}

/** Jev picks a skill (and how much) from the wrist picture each step; code runs it and shows Jev the result next step. */
export interface Helpers {
	/** Claude, asked at checkpoints: after a skill with a checkpoint succeeds, when Jev says done, and at a handback. */
	overseer?: Overseer;
	/** The words Jev reads, with Claude's rewrites; read afresh each step. */
	prompts?: () => Prompts;
}

export async function runGoal(goal: Goal, deps: ProgramDeps, signal: AbortSignal, skills: readonly Skill[] = SKILLS, scratch: Scratch = {}, more: Helpers = {}): Promise<GoalReport> {
	const prompts = more.prompts ?? (() => ({ texts: {}, history: [] }));
	const names = skills.map((k) => k.name);
	const clash = names.find((n, i) => isEnd(n) || names.indexOf(n) !== i);
	if (clash) throw new Error(`skill name ${clash} is taken (skills: ${names.join(", ")}; done and stuck are reserved)`);
	const id = goal.id ?? "do";
	const max = goal.steps ?? DEFAULT_STEPS;
	const steps: GoalStep[] = [];
	const ran: GoalStep[] = [];
	let unsure = 0;

	let n = 0;
	let decided: Frame | undefined;
	/** Asks Claude about a checkpoint; its verdict joins the recent steps Jev sees. Null when there is no overseer or it failed. */
	const oversee = async (kind: CheckpointKind, reason?: string) => {
		if (!more.overseer) return null;
		try {
			await settle();
			const [arm, now] = await Promise.all([deps.arm.state(), deps.arm.frames(["wrist"])]);
			const frames = [...(decided ? [{ label: "the picture Jev chose its last step on", frame: decided }] : []), ...(now.frames.wrist ? [{ label: "the picture now", frame: now.frames.wrist }] : [])];
			const recent = ran.slice(-RECENT).map((r) => ({ skill: r.skill, ...(r.amount !== undefined ? { amount: r.amount } : {}), ok: r.ok, result: r.why }));
			const facts = factsOf(arm, scratch);
			const ready = readySkills(skills, facts).map((k) => k.name);
			const v = await more.overseer.look({ kind, goal: goal.text, what: goal.what, ...(reason ? { reason } : {}), facts, recent, texts: currentTexts(skills, prompts()), ready, frames });
			const note: GoalStep = { skill: `claude:${kind}`, p: 1, ok: v.agree, why: `Claude ${v.agree ? "agrees" : "disagrees"}: ${v.why}${v.edited.length ? ` (rewrote ${v.edited.join(", ")})` : ""}`, ms: 0 };
			ran.push(note);
			await record(n, note);
			return v;
		} catch (e) {
			await record(n, { skill: `claude:${kind}`, p: 0, ok: false, why: `Claude could not be asked: ${(e as Error).message}`, ms: 0 });
			return null;
		}
	};
	const end = async (exit: Exit, reason: string): Promise<GoalReport> => {
		if (exit === "interrupted") await deps.arm.hold().catch(() => undefined);
		return { exit, reason, steps };
	};
	let replans = 0;
	/** Runs a skill, records it, and asks Claude at its checkpoint; returns a report when the goal ends on it. */
	const runStep = async (skill: Skill, how: { amount?: number }, p: number, note: string): Promise<GoalReport | null> => {
		const run = await runSkill(skill, { what: goal.what, ...how, ...(more.overseer ? { oversee: (kind: "aligned" | "gripped") => oversee(kind) } : {}) }, scratch, deps, signal, `${id}.${n}`);
		const same = run.result.ok ? 0 : ran.slice(-(REPEATS - 1)).filter((r) => r.skill === skill.name && !r.ok && r.why.startsWith(run.result.why)).length;
		const step: GoalStep = { skill: skill.name, ...how, p, ok: run.result.ok, why: `${run.result.why}${note}`, ms: run.ms };
		ran.push(step);
		await record(n, step, run);
		if (same >= REPEATS - 1) return handback("stuck", `${skill.name} failed the same way ${REPEATS} times in a row: ${run.result.why}`);
		if (run.result.ok && skill.checkpoint && !run.result.claude_checked) {
			const v = await oversee(skill.checkpoint.kind);
			if (v && !v.agree) skill.checkpoint.refute(scratch);
		}
		return null;
	};
	/**
	 * Jev stopped short of the goal: Claude looks, may rewrite Jev's words, and may name the step to take instead of
	 * stopping. Code runs that step if its precondition holds, at most MAX_REPLANS times a goal.
	 */
	const handback = async (exit: Exit, reason: string): Promise<GoalReport | null> => {
		const v = await oversee("handback", reason);
		if (!v?.next || replans >= MAX_REPLANS || signal.aborted) return end(exit, reason);
		const skill = applyPrompts(skills, prompts()).find((k) => k.name === v.next!.skill);
		const ready = skill && (!skill.ready || skill.ready(factsOf(await deps.arm.state(), scratch)));
		if (!skill || !ready) {
			await record(n, { skill: `claude:next`, p: 0, ok: false, why: `Claude named ${v.next.skill}, which ${skill ? "cannot run now" : "is no skill"}`, ms: 0 });
			return end(exit, reason);
		}
		replans++;
		unsure = 0;
		const how = skill.amount && v.next.amount !== undefined ? { amount: v.next.amount } : {};
		return runStep(skill, how, 1, " (Claude's choice after Jev stopped)");
	};
	const record = async (n: number, step: GoalStep, run?: SkillRun) => {
		steps.push(step);
		const skill = skills.find((k) => k.name === step.skill);
		await deps.recorder.step({
			task: id,
			step: n,
			goal: goal.text,
			what: goal.what,
			skill: step.skill,
			amount: step.amount ?? null,
			unit: (step.amount !== undefined && skill?.amount?.unit) || null,
			p: step.p,
			ok: step.ok,
			why: step.why,
			result: deps.recorder.redact(run?.result ?? null),
			log: run?.log ?? [],
			held: deps.recorder.redact(scratch.held ?? null),
			ms: step.ms,
		});
	};

	try {
		await deps.arm.hold();
	} catch (e) {
		return end("failed", `armd would not hold: ${(e as Error).message}`);
	}
	for (n = 1; n <= max; n++) {
		if (signal.aborted) return end("interrupted", "stopped from outside");
		let reading: GoalReading | undefined;
		let only = false;
		try {
			if (n > 1) await settle();
			const t0 = performance.now();
			const [arm, look] = await Promise.all([deps.arm.state(), deps.arm.frames(["wrist"], deps.model.caps.maxSide)]);
			const framesMs = Math.round(performance.now() - t0);
			if (!look.frames.wrist) return end("failed", `no wrist frame (${look.missing.wrist ?? "missing"})`);
			const recent = ran.slice(-RECENT);
			const facts = factsOf(arm, scratch);
			const words = prompts();
			const offered = applyPrompts(readySkills(skills, facts), words);
			decided = look.frames.wrist;
			if (offered.length === 1 && !offered[0]!.amount) {
				reading = { choice: offered[0]!.name, p: 1, margin: 1, skill: offered[0]! };
				only = true;
			} else await deps.model.decide(compileGoal(goal, offered, { arm, frame: look.frames.wrist, held: scratch.held, facts, recent, view: deps.view }, currentTexts(skills, words).choice), {
				role: "step",
				task: id,
				tick: n,
				facts: { arm, recent, framesMs },
				plan: (a): DecisionPlan => {
					reading = readGoal(a, offered);
					return { reading: { action: reading.choice, p: reading.p }, motion: null, step: amountText(reading.skill, reading.amount) };
				},
			});
		} catch (e) {
			return end("failed", `step ${n}: ${(e as Error).message}`);
		}
		const { choice, p, skill, amount } = reading!;
		const how = amount !== undefined ? { amount } : {};

		if (p < SKILL_SURE || reading!.margin < SKILL_MARGIN) {
			await record(n, { skill: choice, ...how, p, ok: false, why: "unsure; ran nothing", ms: 0 });
			if (++unsure >= UNSURE_HANDBACK) {
				const r = await handback("unsure", `Jev was unsure ${unsure} steps in a row (last: ${choice} at p ${p.toFixed(2)})`);
				if (r) return r;
			}
			continue;
		}
		unsure = 0;
		if (!skill) {
			await record(n, { skill: choice, p, ok: choice === "done", why: choice === "done" ? "Jev judged the goal accomplished" : "Jev judged the goal out of reach of these skills", ms: 0 });
			if (choice === "done") {
				const v = await oversee("done");
				if (v && !v.agree) continue;
				return end("done", `Jev chose done (p ${p.toFixed(2)})${v ? "; Claude agreed" : ""}`);
			}
			const r = await handback("stuck", `Jev chose stuck (p ${p.toFixed(2)})`);
			if (r) return r;
			continue;
		}
		const r = await runStep(skill, how, p, only ? " (the only skill whose preconditions held; Jev was not asked)" : "");
		if (r) return r;
	}
	await oversee("handback", `${max} steps taken`);
	return end("budget", `${max} steps taken`);
}
