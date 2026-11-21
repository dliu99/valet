const W = (await import("node:url")).fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
const { openStore, ctx, RunDoc } = await import(`${W}/brain/record.ts`);
process.chdir(W);
const store = await openStore("runs", { readOnly: true });
const run = await store.harness.snapshot(RunDoc, Number(process.argv[2]), ctx);
for (const c of run.checkpoints ?? []) console.log(c.n, c.kind, c.agree, (c.frames ?? []).join(","), (c.move ? JSON.stringify(c.move) : ""), c.why.slice(0, 120));
await store.close();
