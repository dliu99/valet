for (const pan of [-30, 0, 30]) {
  await moveTo({ pan_deg: pan, reach_cm: 16, height_cm: 22, pitch_deg: 40 });
  await sleep(300);
  const f = await locate("white supplement bottle");
  const b = await locate("white box");
  log(`pan ${pan}: bottle ${f.found ? JSON.stringify(f.box) + " foot " + JSON.stringify(f.foot) : "-"} | box ${b.found ? JSON.stringify(b.box) + " foot " + JSON.stringify(b.foot) : "-"}`);
}
