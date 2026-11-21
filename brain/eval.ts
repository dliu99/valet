#!/usr/bin/env node
/**
 * Offline eval of Jev's skill choice in `valet do`, without the arm. Each line of runs/eval/steps.jsonl is one recorded
 * step decision (its wrist frame, facts, recent steps) with the skills a careful operator would accept next, judged by
 * looking at the frame. Every decision is asked again with today's skills (offered by their `ready` on the recorded
 * facts) and today's wording (runs/prompts.json, plus any variant being tried), and scored the way the director acts.
 *
 *   npm run eval -- [--variants <file.json>] [--repeat N] [--only <id,id>] [--clear] [--save <variant>]
 *
 * A variants file maps a name to the texts it changes, by prompt target (`choice`, a skill's name, `<skill>.amount`):
 * { "short_descend": { "descend": "..." } }. The current wording always runs first, as `current`.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import type { ArmState, Frame } from "./arm.ts";
import { DECIDER_KINDS, loadConfig, loadEnv, type DeciderKind } from "./config.ts";
import { makeDecider, topChoice, type Answers, type DecisionRequest } from "./decision.ts";
import { CHOICE_ID, compileGoal, readySkills, SKILL_MARGIN, SKILL_SURE, type GoalStep } from "./director.ts";
import { applyPrompts, CHOICE_TARGET, currentTexts, readPrompts, savePromptEdits, type Prompts } from "./prompts.ts";
import { fileSink, recorded, Recorder, type RecordedModel } from "./record.ts";
import { GRASP, SKILLS, type Facts, type Skill } from "./skills.ts";

/** One recorded step decision, labelled. */
export interface Labelled {
	/** `<run>.<tick>` */
	id: string;
	run: number;
	tick: number;
	frame: { sha256: string; width: number; height: number };
	state: { goal: string; what: string; held: string | null; arm: ArmState; facts: Facts; recent: GoalStep[] };
	/** What the run offered Jev then, and what Jev picked; for reference only. */
	offered_then: string[];
	jev_then: { choice: string | null; p: number | null };
	/** Skills (or done, stuck) a careful operator would accept as the next step. */
	accepted: string[];
	/** The picture and the facts leave room for doubt (a stale frame, facts that contradict the picture, a close call). */
	ambiguous: boolean;
	reason: string;
}

export interface Scored {
	id: string;
	phase: string;
	ambiguous: boolean;
	offered: string[];
	/** The accepted choices that are on offer now. */
	wanted: string[];
	choice: string;
	p: number;
	margin: number;
	/** The director would run it: p at least SKILL_SURE and SKILL_MARGIN ahead of the runner-up. */
	acted: boolean;
	right: boolean;
	/** Only one skill was ready, so the director would run it without asking. */
	only?: boolean;
}

/** Where the arm is in the pick, from the facts alone: the confusion summary groups by this. */
export function phaseOf(f: Facts): string {
	if (f.object_held) return f.at_set_down_height ? "held, at set-down height" : f.over_set_down_spot ? "held, over the set-down spot" : "held, carrying";
	if (f.gripper === "holding") return f.grip_failed_check ? "gripping, failed its check" : "gripping, not yet lifted";
	if (f.object_set_down) return "set down";
	const shut = f.gripper !== "open" ? ", jaws shut" : "";
	if (!f.object_located) return `nothing located${shut}`;
	return f.jaw_tip_height_cm >= GRASP.clear_cm ? `located, jaws above${shut}` : `located, jaws low${shut}`;
}

const ENDS = ["done", "stuck"];

/** The skills Jev would be offered now on the recorded facts, worded as `prompts` says. */
export function offeredFor(item: Labelled, skills: readonly Skill[], prompts: Prompts): Skill[] {
	return applyPrompts(readySkills(skills, item.state.facts), prompts);
}

/** The request the director would send now for this recorded situation. */
export function requestFor(item: Labelled, skills: readonly Skill[], prompts: Prompts, frame: Frame, view: string): { request: DecisionRequest; offered: Skill[] } {
	const offered = offeredFor(item, skills, prompts);
	const s = item.state;
	const held = s.held ? { what: s.held, grasp: { pan_deg: 0, reach_cm: 0, height_cm: 0 } } : undefined;
	const request = compileGoal({ text: s.goal, what: s.what }, offered, { arm: s.arm, frame, held, facts: s.facts, recent: s.recent, view }, currentTexts(skills, prompts)[CHOICE_TARGET]);
	return { request, offered };
}

/** Scores Jev's answer against the labels, the way the director reads it. */
export function score(item: Labelled, offered: readonly string[], answers: Answers): Scored {
	const a = answers[CHOICE_ID];
	const ranked = a?.type === "choice" ? Object.values(a.probabilities ?? {}).sort((x, y) => y - x) : [];
	const top = topChoice(a) ?? { choice: "(none)", p: 0 };
	const margin = ranked.length > 1 ? ranked[0]! - ranked[1]! : 1;
	const names = [...offered, ...ENDS];
	const wanted = item.accepted.filter((c) => names.includes(c));
	return {
		id: item.id,
		phase: phaseOf(item.state.facts),
		ambiguous: item.ambiguous,
		offered: [...offered],
		wanted,
		choice: top.choice,
		p: top.p,
		margin: Math.round(margin * 1000) / 1000,
		acted: top.p >= SKILL_SURE && margin >= SKILL_MARGIN,
		right: wanted.includes(top.choice),
	};
}

export interface Confusion {
	phase: string;
	wanted: string;
	picked: string;
	n: number;
	/** How many of them the director would have acted on. */
	acted: number;
	ids: string[];
}

export interface Summary {
	/** How many times each decision was asked; the counts below add up over all passes. */
	passes: number;
	n: number;
	right: number;
	acted: number;
	actedRight: number;
	actedWrong: number;
	unsure: number;
	/** Top choice among the accepted ones. */
	accuracy: number;
	/** Right among the choices the director would act on. */
	actedAccuracy: number;
	unsureRate: number;
	confusions: Confusion[];
}

const ratio = (a: number, b: number) => (b ? Math.round((a / b) * 1000) / 1000 : 0);

export function summarize(rows: readonly Scored[], passes = 1): Summary {
	const acted = rows.filter((r) => r.acted);
	const actedRight = acted.filter((r) => r.right).length;
	const right = rows.filter((r) => r.right).length;
	const byKey = new Map<string, Confusion>();
	for (const r of rows) {
		if (r.right) continue;
		const wanted = r.wanted.join(" | ");
		const key = `${r.phase}\u0000${wanted}\u0000${r.choice}`;
		const c = byKey.get(key) ?? { phase: r.phase, wanted, picked: r.choice, n: 0, acted: 0, ids: [] };
		c.n++;
		if (r.acted) c.acted++;
		c.ids.push(r.id);
		byKey.set(key, c);
	}
	return {
		passes,
		n: rows.length,
		right,
		acted: acted.length,
		actedRight,
		actedWrong: acted.length - actedRight,
		unsure: rows.length - acted.length,
		accuracy: ratio(right, rows.length),
		actedAccuracy: ratio(actedRight, acted.length),
		unsureRate: ratio(rows.length - acted.length, rows.length),
		confusions: [...byKey.values()].sort((x, y) => y.n - x.n || y.acted - x.acted),
	};
}

export function readLabelled(file: string): Labelled[] {
	return readFileSync(file, "utf8")
		.split("\n")
		.filter((l) => l.trim())
		.map((l) => JSON.parse(l) as Labelled);
}

/** Runs `fn` over `items`, at most `width` at a time, keeping their order. */
async function pool<T, R>(items: readonly T[], width: number, fn: (item: T, i: number) => Promise<R>): Promise<R[]> {
	const out: R[] = new Array(items.length);
	let next = 0;
	await Promise.all(
		Array.from({ length: Math.min(width, items.length) }, async () => {
			while (next < items.length) {
				const i = next++;
				out[i] = await fn(items[i]!, i);
			}
		}),
	);
	return out;
}

interface Asker {
	model: RecordedModel;
	recorder: Recorder;
	view: string;
}

/** Asks Jev every labelled decision once with these texts; the ones with nothing accepted on offer are left out. */
export async function evaluate(items: readonly Labelled[], skills: readonly Skill[], prompts: Prompts, ask: Asker, label: string, width = 4): Promise<{ rows: Scored[]; skipped: string[] }> {
	const skipped: string[] = [];
	const rows = await pool(items, width, async (item): Promise<Scored | null> => {
		const frame = ask.recorder.loadFrame(item.frame.sha256, "wrist", item.frame);
		if (!frame) throw new Error(`${item.id}: no frame ${item.frame.sha256} in ${ask.recorder.framesDir}`);
		const { request, offered } = requestFor(item, skills, prompts, frame, ask.view);
		const names = offered.map((k) => k.name);
		if (!item.accepted.some((c) => names.includes(c) || ENDS.includes(c))) {
			skipped.push(item.id);
			return null;
		}
		if (offered.length === 1 && !offered[0]!.amount) {
			const choice = offered[0]!.name;
			return { ...score(item, names, { [CHOICE_ID]: { type: "choice", choice, probabilities: { [choice]: 1 }, confidence: 1 } }), only: true };
		}
		const answers = await ask.model.decide(request, { role: "replay", task: `eval:${label}:${item.id}`, tick: item.tick, facts: { arm: item.state.arm, recent: item.state.recent } });
		return score(item, names, answers);
	});
	return { rows: rows.filter((r): r is Scored => r !== null), skipped };
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

/** A count per pass. */
const per = (s: Summary, count: number) => (s.passes === 1 ? String(count) : (count / s.passes).toFixed(1));

export function table(results: readonly { name: string; summary: Summary }[]): string {
	const head = ["variant", "n", "top-1", "acted", "acted right", "acted wrong", "unsure"];
	const lines = results.map(({ name, summary: s }) => [name, per(s, s.n), pct(s.accuracy), per(s, s.acted), `${per(s, s.actedRight)} (${pct(s.actedAccuracy)})`, per(s, s.actedWrong), pct(s.unsureRate)]);
	const widths = head.map((h, i) => Math.max(h.length, ...lines.map((l) => l[i]!.length)));
	return [head, ...lines].map((l) => l.map((c, i) => c.padEnd(widths[i]!)).join("  ")).join("\n");
}

export function confusionText(s: Summary, top = 10): string {
	return s.confusions
		.slice(0, top)
		.map((c) => `  ${String(c.n).padStart(3)}x (${c.acted} acted)  ${c.phase}: wanted ${c.wanted || "-"}, picked ${c.picked}   e.g. ${c.ids.slice(0, 4).join(", ")}`)
		.join("\n");
}

async function main(): Promise<number> {
	const { values } = parseArgs({
		options: {
			set: { type: "string" },
			variants: { type: "string" },
			only: { type: "string" },
			clear: { type: "boolean" },
			save: { type: "string" },
			decider: { type: "string" },
			width: { type: "string" },
			config: { type: "string" },
			runs: { type: "string" },
			repeat: { type: "string" },
		},
	});
	loadEnv();
	const config = loadConfig(values.config);
	const runsDir = values.runs ?? config.runsDir;
	const kind = (values.decider ?? "jev") as DeciderKind;
	if (!DECIDER_KINDS.includes(kind)) throw new Error(`--decider must be one of ${DECIDER_KINDS.join(", ")}`);
	let items = readLabelled(values.set ?? join(runsDir, "eval", "steps.jsonl"));
	if (values.only) items = items.filter((i) => values.only!.split(",").includes(i.id));
	if (values.clear) items = items.filter((i) => !i.ambiguous);
	const base = readPrompts(runsDir);
	const variants: Record<string, Record<string, string>> = { current: {}, ...(values.variants ? JSON.parse(readFileSync(values.variants, "utf8")) : {}) };
	const targets = currentTexts(SKILLS, base);
	for (const [name, texts] of Object.entries(variants))
		for (const t of Object.keys(texts)) if (!(t in targets)) throw new Error(`variant ${name}: no prompt target ${t} (targets: ${Object.keys(targets).join(", ")})`);

	const stamp = new Date().toISOString().replace(/[:.]/g, "-");
	const outDir = join(runsDir, "eval");
	mkdirSync(join(outDir, "log"), { recursive: true });
	const recorder = new Recorder(join(runsDir, "frames"), fileSink(join(outDir, "log", `${stamp}.jsonl`)));
	const decider = await makeDecider(kind, { config });
	const ask: Asker = { model: recorded(decider, recorder), recorder, view: config.viewpoints.wrist };
	const width = Number(values.width ?? 4);
	const passes = Math.max(1, Number(values.repeat ?? 1));

	const results: { name: string; texts: Record<string, string>; summary: Summary; rows: Scored[]; skipped: string[] }[] = [];
	for (const [name, texts] of Object.entries(variants)) {
		const prompts: Prompts = { texts: { ...base.texts, ...texts }, history: base.history };
		const t0 = performance.now();
		const rows: Scored[] = [];
		let skipped: string[] = [];
		for (let pass = 0; pass < passes; pass++) {
			const done = await evaluate(items, SKILLS, prompts, ask, name, width);
			rows.push(...done.rows);
			skipped = done.skipped;
		}
		const summary = summarize(rows, passes);
		results.push({ name, texts, summary, rows, skipped });
		console.log(`${name}: ${rows.length} asked in ${((performance.now() - t0) / 1000).toFixed(1)} s${skipped.length ? `; ${skipped.length} with nothing accepted on offer now: ${skipped.join(", ")}` : ""}`);
		console.log(confusionText(summary));
	}
	console.log(`\n${table(results)}`);
	const clear = results.map((r) => ({ name: r.name, summary: summarize(r.rows.filter((x) => !x.ambiguous), passes) }));
	if (!values.clear && items.some((i) => i.ambiguous)) console.log(`\nwithout the ambiguous ones:\n${table(clear)}`);
	writeFileSync(join(outDir, `results-${stamp}.json`), `${JSON.stringify({ at: stamp, decider: decider.id, results }, null, 1)}\n`);
	console.log(`\nscored rows in ${join(outDir, `results-${stamp}.json`)}; Jev's exchanges in ${join(outDir, "log", `${stamp}.jsonl`)}`);

	if (values.save) {
		const won = results.find((r) => r.name === values.save);
		if (!won || won.name === "current") throw new Error(`--save names a variant from the file, not ${values.save}`);
		const was = results[0]!.summary;
		const now = won.summary;
		const numbers = `acted right ${per(was, was.actedRight)}->${per(now, now.actedRight)}, acted wrong ${per(was, was.actedWrong)}->${per(now, now.actedWrong)}, top-1 ${pct(was.accuracy)}->${pct(now.accuracy)} on ${per(now, now.n)} labelled steps${passes > 1 ? `, mean of ${passes} passes` : ""}`;
		const refused = savePromptEdits(runsDir, SKILLS, Object.entries(won.texts).map(([target, text]) => ({ target, text, reason: `offline eval: ${numbers}` })));
		console.log(refused.length ? `refused: ${refused.join("; ")}` : `saved ${Object.keys(won.texts).join(", ")} to ${join(runsDir, "prompts.json")} (${numbers})`);
	}
	return 0;
}

if (import.meta.main)
	main().then(
		(code) => process.exit(code),
		(e) => {
			console.error(`eval: ${(e as Error).message}`);
			process.exit(1);
		},
	);
