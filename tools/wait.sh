#!/bin/bash
# wait until the latest run has more than N programs or has ended, up to T seconds
cd "$(dirname "$0")/.."
N=$1; T=${2:-300}; end=$((SECONDS+T))
while [ $SECONDS -lt $end ]; do
  out=$(node --input-type=module -e 'const { openStore, ctx, RunDoc, RunIndex } = await import("./brain/record.ts"); const s = await openStore("runs", { readOnly: true }); const root = await s.harness.root(ctx); const id = (await s.harness.snapshot(RunIndex, root.id, ctx)).runs.at(-1).conversationId; const r = await s.harness.snapshot(RunDoc, id, ctx); console.log((r.programs ?? []).length, r.status); await s.close();' 2>/dev/null)
  set -- $out; n=$1; st=$2
  if [ "$st" != "running" ] || [ "$n" -gt "$N" ]; then echo "programs=$n status=$st"; exit 0; fi
  sleep 5
done
echo "timeout programs=$n status=$st"
