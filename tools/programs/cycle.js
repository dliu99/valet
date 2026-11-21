const what = "white supplement bottle";
for (const side of ["place_right", "place_left", "place_right", "place_left"]) {
  const t0 = Date.now();
  const p = await skill("pick", { what });
  log(`CYCLE pick ${p.ok ? "ok" : "FAIL"} ${p.why} (${p.tries ?? "-"} tries)`);
  if (!p.ok) break;
  const d = await skill(side, { what, amount: 10 });
  log(`CYCLE ${side} ${d.ok ? "ok" : "FAIL"} ${d.why} [${Math.round((Date.now() - t0) / 1000)} s]`);
  if (!d.ok) break;
}
