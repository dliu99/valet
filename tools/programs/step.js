log(await moveTo({ reach_cm: 22, height_cm: 10 }));
log(await moveTo({ height_cm: 4.8 }));
log(await grip("open", 85));
log(await move({ height_cm: 8 }));
log(await move({ reach_cm: -4 }));
await sleep(800);
log("upright?", await check("Is the white box standing upright on the desk?"));
