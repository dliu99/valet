import assert from "node:assert/strict";
import { test } from "node:test";
import { createModels } from "@earendil-works/pi-ai/models";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { createRegistry, ToolResultEntry, type ConversationId } from "@earendil-works/pi-durable";
import { armClient } from "../arm.ts";
import { DEFAULTS } from "../config.ts";
import { plannerExtension, planRun, type PlannerServices } from "../planner.ts";
import { ctx, LlmRequestEntry, openStore, type Store } from "../record.ts";
import { startArmdDouble } from "./armd-double.ts";
import { choose, scriptedDecider, tempDir } from "./helpers.ts";

async function plannerRig() {
	const armd = await startArmdDouble();
	const arm = armClient(armd.url);
	await arm.act({ kind: "remember", name: "overview" });
	armd.calls.length = 0;
	const models = createModels();
	const registry = createRegistry();
	const store = await openStore(tempDir(), { models, registry });
	const svc: PlannerServices = { config: { ...DEFAULTS, armdUrl: armd.url }, arm, decider: scriptedDecider((req, tick) => choose(req, tick === 1 ? "open" : "done")), store, person: async () => "no one is here" };
	registry.install(plannerExtension(svc));
	return { armd, store, svc, models, close: async () => (await store.close(), await armd.close()) };
}

async function entries(store: Store, id: ConversationId) {
	const conv = (await store.harness.conversation(id, ctx))!;
	return (await conv.entries({ order: "ascending" }, 2000, undefined, ctx)).items;
}

test("program runs Claude's code on the arm and returns the log, the pictures and a record", async () => {
	const faux = fauxProvider({ provider: "scripted", models: [{ id: "t", input: ["text", "image"] }] });
	const code = 'const a = await arm(); log("start", a.grip); const m = await move({ height_cm: 2 }); log(m.result); const g = await grip("close"); log(g.grip);';
	faux.setResponses([
		fauxAssistantMessage(fauxToolCall("program", { code }), { stopReason: "toolUse" }),
		fauxAssistantMessage(fauxToolCall("finish", { summary: "done", met: true }), { stopReason: "toolUse" }),
	]);
	const r = await plannerRig();
	r.models.setProvider(faux.provider);
	try {
		const { id, run } = await planRun(r.svc, "try the api", { provider: "scripted", modelId: "t" });
		const result = (await entries(r.store, id)).filter((e) => ToolResultEntry.is(e)).map((e) => e.model![0] as any)[0];
		const out = JSON.parse(result.content[0].text);
		assert.equal(out.ok, true, out.error);
		assert.deepEqual(out.log.slice(1), ["moved", "empty"]);
		assert.ok(result.content.some((c: any) => c.type === "image"), "the planner sees the final picture");
		assert.equal(r.armd.calls.filter((c) => c === "POST /act").length, 2);
		assert.equal(run.programs.length, 1);
		assert.equal((run.programs[0] as any).code, code);
	} finally {
		await r.close();
	}
});

test("every planner provider request is recorded, its images as frame hashes", async () => {
	const faux = fauxProvider({ provider: "scripted", models: [{ id: "t", input: ["text", "image"] }] });
	const asProvider = (reply: ReturnType<typeof fauxAssistantMessage>) => async (context: any, options: any, _state: unknown, model: any) => {
		await options?.onPayload?.({ messages: context.messages }, model);
		return reply;
	};
	faux.setResponses([
		asProvider(fauxAssistantMessage(fauxToolCall("look", {}), { stopReason: "toolUse" })),
		asProvider(fauxAssistantMessage(fauxToolCall("finish", { summary: "looked", met: true }), { stopReason: "toolUse" })),
	]);
	const r = await plannerRig();
	r.models.setProvider(faux.provider);
	try {
		const { id, run } = await planRun(r.svc, "look around", { provider: "scripted", modelId: "t" });
		assert.equal(run.status, "done");
		const requests = (await entries(r.store, id)).filter((e) => e.kind === LlmRequestEntry.kind).map((e) => e.data as any);
		assert.equal(requests.length, 2);
		const system = requests[0].body.messages.find((m: any) => m.role === "system");
		assert.ok(system.sections.preamble !== undefined || system.sections.jev, "the system prompt as sent, with its sections");
		const second = JSON.stringify(requests[1].body);
		const look = requests[1].body.messages.find((m: any) => m.role === "toolResult");
		const image = look.content.find((c: any) => c.type === "image");
		assert.match(image.data, /^[0-9a-f]{64}$/, "the look image is kept as its frame hash");
		assert.ok(!/[A-Za-z0-9+/]{1000,}/.test(second), "no image bytes in the record");
	} finally {
		await r.close();
	}
});

test("finish keeps a lesson and a met run's programs for the next run's prompt", async () => {
	const { readMemory } = await import("../planner.ts");
	const faux = fauxProvider({ provider: "scripted", models: [{ id: "t", input: ["text", "image"] }] });
	faux.setResponses([
		fauxAssistantMessage(fauxToolCall("program", { code: 'log("hi")' }), { stopReason: "toolUse" }),
		fauxAssistantMessage(fauxToolCall("finish", { summary: "done", met: true, lesson: "Grip 2 cm below the cap." }), { stopReason: "toolUse" }),
	]);
	const r = await plannerRig();
	r.models.setProvider(faux.provider);
	try {
		await planRun(r.svc, "say hi", { provider: "scripted", modelId: "t" });
		const memory = readMemory(r.store.runsDir);
		assert.deepEqual(memory.lessons, ["(met) Grip 2 cm below the cap."]);
		assert.deepEqual(memory.skill, { goal: "say hi", programs: ['log("hi")'] });
	} finally {
		await r.close();
	}
});
