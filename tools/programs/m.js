log(await grip("open", 70));
log(await move({ height_cm: 5 }));
log(await moveTo({ pan_deg: 8, reach_cm: 26, height_cm: 16, pitch_deg: 50 }));
log(await measure("white supplement bottle"));
