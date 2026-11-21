log(await arm());
await grip("open");
log(await moveTo({ pan_deg: 20, reach_cm: 18, height_cm: 14, pitch_deg: 48 }));
log(await locate("white supplement bottle with a blue cap"));
