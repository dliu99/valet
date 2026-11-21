import assert from "node:assert/strict";
import { test } from "node:test";
import { createModels } from "@earendil-works/pi-ai/models";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { createRegistry } from "@earendil-works/pi-durable";
import { overseerExtension, startOverseer } from "../overseer.ts";
import { readPrompts } from "../prompts.ts";
import { conversationSink, ctx, openStore, Recorder, RunDoc, startRun, type RunState } from "../record.ts";
import { SKILLS } from "../skills.ts";
import { tempDir } from "./helpers.ts";

const frame = { cam: "wrist", sha256: "", width: 2, height: 2, mime: "image/jpeg", data_b64: Buffer.from("not really a jpeg").toString("base64"), t: 0 } as never;

test("Claude's verdict at a checkpoint comes back to code, is recorded, and its rewrites reach the words Jev reads", async () => {
	const faux = fauxProvider({ provider: "scripted", models: [{ id: "t", input: ["text", "image"] }] });
	const edit = { target: "close", text: "Close only when the object sits between the two jaws.", reason: "it closed on air" };
	let sent: any;
	faux.setResponses([
		async (context: any) => {
			sent = context.messages.at(-1);
			return fauxAssistantMessage(fauxToolCall("verdict", { agree: false, why: "the jaws are left of the bottle", edits: [edit] }), { stopReason: "toolUse" });
		},
	] as never);
	const models = createModels();
	models.setProvider(faux.provider);
	const registry = createRegistry();
	const store = await openStore(tempDir(), { models, registry });
	registry.install(overseerExtension({ store, skills: SKILLS }));
	try {
		const id = await startRun(store, { kind: "do", goal: "g", planner: "scripted/t", decider: "jev" });
		const overseer = await startOverseer(store, id, { provider: "scripted", modelId: "t" }, new Recorder(store.framesDir, conversationSink(store, id)));
		const v = await overseer.look({ kind: "gripped", goal: "move the bottle", what: "bottle", facts: { object_held: true }, recent: [], texts: { close: "Close the jaws on the object." }, frames: [{ label: "the picture now", frame }] });
		assert.deepEqual(v, { agree: false, why: "the jaws are left of the bottle", edited: ["close"], refused: [] });
		assert.ok(sent.content.some((c: any) => c.type === "image"), "Claude is shown the picture");
		assert.match(sent.content[0].text, /"checkpoint": "gripped"/);
		const run = (await store.harness.snapshot(RunDoc, id, ctx)) as RunState;
		assert.equal((run.checkpoints[0] as any).kind, "gripped");
		assert.equal(readPrompts(store.runsDir).texts.close, edit.text);
	} finally {
		await store.close();
	}
});
