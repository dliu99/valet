const W = (await import("node:url")).fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
const { openStore, ctx, RunDoc, RunIndex, runEntries } = await import(`${W}/brain/record.ts`);
process.chdir(W);
const store = await openStore("runs", { readOnly: true });
let id = Number(process.argv[2]);
if (!id) { const r = await store.harness.root(ctx); id = (await store.harness.snapshot(RunIndex, r.id, ctx)).runs.at(-1).conversationId; }
const run = await store.harness.snapshot(RunDoc, id, ctx);
console.log(`run ${id}: ${run.status} ${run.summary ?? ""} (${(run.programs ?? []).length} programs)`);
for (const p of (run.programs ?? []).slice(Number(process.argv[3] ?? 0))) console.log(`\n=== ${p.id} ${p.ok ? "ok" : "ERROR " + p.error} ${(p.ms / 1000).toFixed(1)}s\n${p.code}\n--- log\n${(p.log ?? []).join("\n")}`);
if (process.argv.includes("--talk")) for (const e of await runEntries(store, id)) if (e.kind === "pi.assistant") for (const c of e.model?.[0]?.content ?? []) if (c.type === "text" && c.text.trim()) console.log("CLAUDE:", c.text.trim().slice(0, 800));
await store.close();
