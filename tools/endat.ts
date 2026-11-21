const W = (await import("node:url")).fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
const { openStore, ctx, RunDoc } = await import(`${W}/brain/record.ts`);
process.chdir(W);
const store = await openStore("runs", { readOnly: true });
const run = await store.harness.snapshot(RunDoc, Number(process.argv[2]), ctx);
console.log(run.endedAt);
await store.close();
