log(await moveTo({ pan_deg: 0, reach_cm: 16, height_cm: 22, pitch_deg: 40 }));
const p = await skill("pick", { what: "white supplement bottle" });
log("PICK", p);
