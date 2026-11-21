for (const pan of [-35, -15, 5, 25, 45]) {
  await moveTo({ pan_deg: pan, reach_cm: 16, height_cm: 22, pitch_deg: 40 });
  await sleep(700);
  const out = [];
  for (const what of ["white supplement bottle with a blue cap", "clear jar with a silver lid", "white box"]) {
    const s = await segment(what, { height_cm: 4 });
    if (s.found && s.score > 0.3) out.push(`${what}: score ${s.score?.toFixed?.(2)} whole ${s.whole} centre ${JSON.stringify(s.centre)}`);
  }
  log(`pan ${pan}: ${out.join(" | ") || "-"}`);
}
