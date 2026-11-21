for (const what of ["white supplement bottle", "white supplement bottle itself, not its shadow on the desk", "blue cap on top of the bottle", "white supplement bottle with a blue cap (the object only, excluding any shadow)"]) {
  const r = await locate(what);
  log(what, "->", r.confidence, JSON.stringify(r.box));
}
