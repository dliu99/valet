log(await moveTo({ pan_deg: 30, reach_cm: 16, height_cm: 22, pitch_deg: 40 }));
const t0 = Date.now();
const s = await segment("white supplement bottle", { height_cm: 5 });
log("sam", Date.now() - t0, "ms", JSON.stringify({ found: s.found, whole: s.whole, box: s.box, foot: s.foot, centre: s.centre }));
log(await find("white supplement bottle"));
