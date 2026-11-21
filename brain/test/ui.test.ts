import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { createModels } from "@earendil-works/pi-ai/models";
import { DEFAULTS } from "../config.ts";
import { CLAIMS, ON_DISAGREE } from "../overseer.ts";
import { DEFAULT_CHOICE, savePromptEdits } from "../prompts.ts";
import { openStore } from "../record.ts";
import { SKILLS } from "../skills.ts";
import { serveUI } from "../ui.ts";
import { startArmdDouble } from "./armd-double.ts";
import { tempDir } from "./helpers.ts";

const JPEG = readFileSync(new URL("./armd-double.jpg", import.meta.url));

async function ui(armdUrl: string, runsDir = tempDir()) {
	const store = await openStore(runsDir, { readOnly: true });
	const server = await serveUI({ store, config: { ...DEFAULTS, armdUrl }, models: createModels(), port: 0 });
	const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	return {
		get: (path: string) => fetch(base + path),
		close: async () => {
			server.closeAllConnections();
			await new Promise((resolve) => server.close(resolve));
			await store.close();
		},
	};
}

test("the live pane reads armd's state and newest wrist frame, and never moves the arm", async () => {
	const armd = await startArmdDouble();
	const page = await ui(armd.url);
	try {
		armd.calls.length = 0;
		const live = (await (await page.get("/api/live")).json()) as { armd: string; state: { kind: string; reach_m: number } };
		assert.equal(live.armd, armd.url);
		assert.equal(live.state.kind, "fake");
		const frame = await page.get("/api/live/wrist.jpg");
		assert.equal(frame.status, 200);
		assert.equal(frame.headers.get("content-type"), "image/jpeg");
		assert.deepEqual(Buffer.from(await frame.arrayBuffer()), JPEG);
		assert.deepEqual(armd.calls, ["GET /state", "POST /frames"]);
	} finally {
		await page.close();
		await armd.close();
	}
});

test("with armd down the live pane says so instead of failing the page", async () => {
	const armd = await startArmdDouble();
	const url = armd.url;
	await armd.close();
	const page = await ui(url);
	try {
		const live = (await (await page.get("/api/live")).json()) as { armd: string; error?: string; state?: unknown };
		assert.equal(live.state, undefined);
		assert.match(live.error ?? "", /unreachable/);
		const frame = await page.get("/api/live/wrist.jpg");
		assert.equal(frame.status, 502);
		assert.match(((await frame.json()) as { error: string }).error, /unreachable/);
	} finally {
		await page.close();
	}
});

test("the skill library is served as Jev reads it, each precondition in terms of the facts", async () => {
	const page = await ui("http://127.0.0.1:1");
	try {
		const lib = (await (await page.get("/api/skills")).json()) as { error: string | null; skills: { name: string; says: string; amount: { levels: number[] } | null; ready: string | null }[] };
		assert.equal(lib.error, null);
		assert.deepEqual(lib.skills.map((k) => k.name), SKILLS.map((k) => k.name));
		for (const [i, k] of lib.skills.entries()) {
			const skill = SKILLS[i]!;
			assert.equal(k.says, skill.says);
			assert.deepEqual(k.amount?.levels ?? null, skill.amount ? [...skill.amount.levels] : null);
			assert.equal(k.ready === null, !skill.ready, `${k.name}: a precondition exactly when the skill has one`);
			const arrow = /^\(?\s*(\w+)\s*\)?\s*=>/.exec(skill.ready?.toString() ?? "");
			if (k.ready && arrow) assert.doesNotMatch(k.ready, new RegExp(`=>|\\b${arrow[1]}\\.`), `${k.name}: ${k.ready}`);
		}
	} finally {
		await page.close();
	}
});

test("the page is given Claude's checkpoint claims and its rewrites of the words Jev reads", async () => {
	const runs = tempDir();
	const close = SKILLS.find((k) => k.name === "close")!;
	savePromptEdits(runs, SKILLS, [{ target: "close", text: "Close only with the object between the jaws.", reason: "it closed on air", run: 7, checkpoint: "gripped" }]);
	const page = await ui("http://127.0.0.1:1", runs);
	try {
		const config = (await (await page.get("/api/config")).json()) as { claims: object; onDisagree: object };
		assert.deepEqual([config.claims, config.onDisagree], [CLAIMS, ON_DISAGREE]);
		const lib = (await (await page.get("/api/skills")).json()) as { prompts: { texts: Record<string, string>; history: { target: string; was: string; run: number; checkpoint: string }[] }; defaults: { choice: string } };
		assert.equal(lib.prompts.texts.close, "Close only with the object between the jaws.");
		assert.deepEqual(lib.prompts.history.map((e) => [e.target, e.was, e.run, e.checkpoint]), [["close", close.says, 7, "gripped"]]);
		assert.equal(lib.defaults.choice, DEFAULT_CHOICE);
	} finally {
		await page.close();
	}
});
