import assert from "node:assert/strict";
import { test } from "node:test";
import { applyPrompts, currentTexts, DEFAULT_CHOICE, readPrompts, savePromptEdits } from "../prompts.ts";
import type { Skill } from "../skills.ts";
import { tempDir } from "./helpers.ts";

const skills: Skill[] = [
	{ name: "close", says: "Close the jaws.", run: async () => ({ ok: true, why: "" }) },
	{ name: "nudge_left", says: "Move left.", amount: { says: "How far?", unit: "cm", levels: [1, 2] }, run: async () => ({ ok: true, why: "" }) },
];

test("rewrites apply to known targets, persist, and keep what they replaced", () => {
	const dir = tempDir();
	const refused = savePromptEdits(dir, skills, [
		{ target: "close", text: "Close only when the object sits between the jaws.", reason: "closed on air" },
		{ target: "nudge_left.amount", text: "How far left, in cm?", reason: "r" },
		{ target: "fly", text: "x", reason: "r" },
		{ target: "choice", text: "", reason: "r" },
	]);
	assert.equal(refused.length, 2);
	const p = readPrompts(dir);
	assert.equal(p.history.length, 2);
	assert.equal(p.history[0]!.was, "Close the jaws.");
	const shown = applyPrompts(skills, p);
	assert.equal(shown[0]!.says, "Close only when the object sits between the jaws.");
	assert.equal(shown[1]!.amount!.says, "How far left, in cm?");
	assert.equal(currentTexts(skills, p).choice, DEFAULT_CHOICE);
});
