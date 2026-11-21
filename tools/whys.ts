const W = (await import("node:url")).fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
const { openStore, ctx, RunDoc } = await import(`${W}/brain/record.ts`);
process.chdir(W);
const store = await openStore("runs", { readOnly: true });
for (const id of process.argv.slice(2).map(Number)) {
  const run = await store.harness.snapshot(RunDoc, id, ctx);
  console.log(`== ${id} ${run.goal}`);
  for (const c of run.checkpoints ?? []) if (["found", "done", "gripped"].includes(c.kind)) console.log(` ${c.kind} ${c.agree}: ${c.why}`);
}
await store.close();
