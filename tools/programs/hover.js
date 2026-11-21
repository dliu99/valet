log(await grip("open", 70));
log(await moveTo({ pan_deg: 12, reach_cm: 24, height_cm: 21, pitch_deg: 45 }));
log(await moveTo({ pan_deg: 15, reach_cm: 27, height_cm: 19, pitch_deg: 50 }));
const r = await locate("white supplement bottle");
log("box", r.confidence, JSON.stringify(r.box));
