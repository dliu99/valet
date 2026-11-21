import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Skill } from "./skills.ts";

/**
 * The words Jev reads, which Claude may rewrite at a checkpoint. A target is `choice` (the question asking for the next
 * skill), a skill's name (what it says it does), or `<skill>.amount` (its how-far question). Edits persist across runs.
 */
export const CHOICE_TARGET = "choice";
export const DEFAULT_CHOICE =
	"Which one skill should the robot arm run next toward `goal`? Judge from `images.wrist` together with `facts` (what the arm's own sensors and earlier looks established) and `recent_steps`.";

export interface PromptEdit {
	target: string;
	text: string;
	reason: string;
	at: number;
	run?: number;
	checkpoint?: string;
	was?: string;
}

export interface Prompts {
	texts: Record<string, string>;
	history: PromptEdit[];
}

const FILE = "prompts.json";
const MAX_TEXT = 400;

export function readPrompts(runsDir: string): Prompts {
	const file = join(runsDir, FILE);
	if (!existsSync(file)) return { texts: {}, history: [] };
	const p = JSON.parse(readFileSync(file, "utf8")) as Partial<Prompts>;
	return { texts: p.texts ?? {}, history: p.history ?? [] };
}

/** Every target and the text Jev reads for it now. */
export function currentTexts(skills: readonly Skill[], p: Prompts): Record<string, string> {
	const out: Record<string, string> = { [CHOICE_TARGET]: p.texts[CHOICE_TARGET] ?? DEFAULT_CHOICE };
	for (const k of skills) {
		out[k.name] = p.texts[k.name] ?? k.says;
		if (k.amount) out[`${k.name}.amount`] = p.texts[`${k.name}.amount`] ?? k.amount.says;
	}
	return out;
}

/** The skills as Jev reads them, with Claude's rewrites in place. */
export function applyPrompts(skills: readonly Skill[], p: Prompts): Skill[] {
	return skills.map((k) => ({
		...k,
		says: p.texts[k.name] ?? k.says,
		...(k.amount ? { amount: { ...k.amount, says: p.texts[`${k.name}.amount`] ?? k.amount.says } } : {}),
	}));
}

/** Applies edits to known targets and saves them; returns why each one that was not applied was refused. */
export function savePromptEdits(runsDir: string, skills: readonly Skill[], edits: readonly Omit<PromptEdit, "at" | "was">[]): string[] {
	const p = readPrompts(runsDir);
	const now = currentTexts(skills, p);
	const refused: string[] = [];
	for (const e of edits) {
		const text = e.text?.trim() ?? "";
		if (!(e.target in now)) refused.push(`${e.target}: no such target (targets: ${Object.keys(now).join(", ")})`);
		else if (!text || text.length > MAX_TEXT) refused.push(`${e.target}: the text must be 1 to ${MAX_TEXT} characters`);
		else {
			p.history.push({ ...e, text, at: Date.now(), was: now[e.target] });
			p.texts[e.target] = text;
			now[e.target] = text;
		}
	}
	writeFileSync(join(runsDir, FILE), `${JSON.stringify(p, null, 1)}\n`);
	return refused;
}
