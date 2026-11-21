await grip("open", 85);
await moveTo({ pan_deg: 10, reach_cm: 18, height_cm: 22, pitch_deg: 40 });
log(await find("white box"));
