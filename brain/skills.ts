import type { Robot } from "./program.ts";

/** A place for the grasp point: pan from the arm's forward axis (positive toward its right), reach from the base axis, height above the table. */
export interface Spot {
	pan_deg: number;
	reach_cm: number;
	height_cm: number;
}

/** What one run's skills know about each other's work. */
export interface Scratch {
	/** Where the object's axis stands, from find_object. */
	target?: Spot;
	/** Where the jaws closed last. */
	closed_at?: Spot;
	/** The object rides in the jaws (checked by sight after lifting), and where it was picked up. */
	held?: { what: string; grasp: Spot };
	/** Where the held object is to be set down. */
	place?: Spot;
	set_down?: boolean;
	/** The tilt approach found reachable above the object. */
	pitch_deg?: number;
	/** Lift found the jaws hold something other than the object. */
	bad_grip?: boolean;
	last?: { skill: string; ok: boolean; why: string };
}

/** What code knows each step, in plain terms: the chooser reads it, and each skill's `ready` is decided on it. */
export interface Facts {
	object_located: boolean;
	jaws_beside_object: boolean;
	jaw_tip_height_cm: number;
	gripper: string;
	object_held: boolean;
	over_set_down_spot: boolean;
	at_set_down_height: boolean;
	object_set_down: boolean;
	/** Lift checked the grip by sight and the object was not in it. */
	grip_failed_check: boolean;
}

const NEAR_CM = 1.5;

/** The jaws just closed on something and lift has not checked it yet: letting go now throws away a grip that may be good. */
const freshGrip = (f: Facts) => f.gripper === "holding" && !f.object_held && !f.grip_failed_check;

/**
 * moveTo, but when the arm cannot hold the asked (or current) tilt there, it goes with the nearest tilt that reaches:
 * far out the arm can only point 40-50° down.
 */
async function goTo(r: Robot, pose: Parameters<Robot["moveTo"]>[0]) {
	try {
		return await r.moveTo(pose);
	} catch (e) {
		const why = String((e as Error).message);
		const can = why.match(/pitches (-?\d+) to (-?\d+)° are reachable/);
		if (!why.startsWith("refused") || !can) throw e;
		const want = pose.pitch_deg ?? (await r.arm()).pitch_deg;
		const pitch_deg = Math.min(Math.max(want, Number(can[1])), Number(can[2]));
		return r.moveTo({ ...pose, pitch_deg });
	}
}

/** move, with goTo's fallback to a tilt the arm can hold at the end point. */
async function goBy(r: Robot, d: { pan_deg?: number; reach_cm?: number; height_cm?: number }) {
	try {
		return await r.move(d);
	} catch (e) {
		if (!String((e as Error).message).startsWith("refused")) throw e;
		const now = await r.arm();
		return goTo(r, { pan_deg: now.pan_deg + (d.pan_deg ?? 0), reach_cm: now.reach_cm + (d.reach_cm ?? 0), height_cm: now.height_cm + (d.height_cm ?? 0) });
	}
}

/**
 * Opens the jaws without dropping anything: if they closed on something and have since risen, they first go back down
 * to where they closed, so whatever they hold is put down rather than let fall.
 */
async function letGo(r: Robot, s: Scratch): Promise<string> {
	const now = await r.arm();
	let put = "";
	if (now.grip !== "open" && s.closed_at && now.height_cm > s.closed_at.height_cm + 1) {
		await goTo(r, { pan_deg: s.closed_at.pan_deg, reach_cm: s.closed_at.reach_cm, height_cm: s.closed_at.height_cm + 0.3 });
		put = "lowered back to where the jaws closed, then ";
	}
	const o = await r.grip("open", GRASP.open_pct);
	s.bad_grip = false;
	return `${put}opened (${o.result})`;
}

export function factsOf(arm: { pan_deg: number; reach_m: number; height_m: number; grip: string }, s: Scratch): Facts {
	const here = { pan_deg: arm.pan_deg, reach_cm: arm.reach_m * 100, height_cm: arm.height_m * 100 };
	const beside = s.target ? besideLeft(s.target, here.height_cm, { pitch_deg: s.pitch_deg ?? GRASP.pitch_deg }) : null;
	const over = Boolean(s.held && s.place && apart(here, s.place) < NEAR_CM);
	return {
		object_located: Boolean(s.target),
		jaws_beside_object: Boolean(beside && !s.held && apart(here, beside) < NEAR_CM),
		jaw_tip_height_cm: r1(here.height_cm),
		gripper: arm.grip,
		object_held: Boolean(s.held),
		over_set_down_spot: over,
		at_set_down_height: Boolean(s.held && here.height_cm < s.held.grasp.height_cm + 1.5),
		object_set_down: Boolean(s.set_down),
		grip_failed_check: Boolean(s.bad_grip),
	};
}

export interface SkillResult {
	ok: boolean;
	why: string;
	[key: string]: unknown;
}

export interface SkillArgs {
	what: string;
	/** The skill's amount, in its unit, when it has one. */
	amount?: number;
	/** Asks Claude about the picture now, when the run has an overseer. */
	oversee?: (kind: "aligned" | "gripped") => Promise<{ agree: boolean; why: string; move?: { right_cm: number; out_cm: number } } | null>;
}

export interface Skill {
	name: string;
	/** One sentence a vision model reads to choose this skill. */
	says: string;
	/** A skill with an amount lets the chooser pick how far, from these levels. */
	amount?: { says: string; unit: "cm" | "deg"; levels: readonly number[] };
	/** Whether the skill can run now; the chooser is offered only skills that can. */
	ready?(f: Facts): boolean;
	/** After it succeeds, Claude checks its claim from the pictures; `refute` undoes the claim when Claude disagrees. */
	checkpoint?: { kind: "found" | "gripped"; refute(s: Scratch): void };
	run(r: Robot, args: SkillArgs, scratch: Scratch): Promise<SkillResult>;
}

const DEG = 180 / Math.PI;
const r1 = (v: number) => Math.round(v * 10) / 10;

export const GRASP = {
	pitch_deg: 55,
	/**
	 * Tilts approach tries in turn. Around 55° the jaws span ~4 cm of depth, enough to absorb find's ±1.5 cm scatter;
	 * near vertical they span ~1 cm and the bottle ends up at the tips (run analysis, 2026-11-20).
	 */
	approach_pitches: [60, 55, 50, 45, 40, 65, 70],
	open_pct: 85,
	/** Grasp-point height when the caller gives none: above the bottle's middle, so it hangs plumb instead of pivoting in the jaws. */
	grip_cm: 4,
	/** Closing and Claude's line-up only make sense with the jaw tips down where they grip: below this. */
	grip_zone_cm: 7,
	/** About the middle of the objects this arm picks, for reading their sides from above. */
	mid_height_cm: 5,
	min_reach_cm: 14,
	/** A view whose centre ray is within about 25° of straight down reads the axis from the box's centre. */
	steep_cos: 0.9,
	/** A box this close to the picture's edge is taken to be cut off by it. */
	edge_margin: 0.04,
	/** How far (in image widths) a held object's box may shift when the arm rises a few cm. */
	held_max_shift: 0.06,
	/** Jev's yes, by eye, that the object is between the jaws, enough to call it held when the box test is unsure. */
	held_seen: 0.6,
	max_reach_cm: 42,
	hover_cm: 13,
	/** Jaw tips at least this high clear the objects this arm picks, so sideways moves there cannot push them. */
	clear_cm: 11,
	/** Rounds of Claude's look-and-move before closing, and the largest move per round. */
	line_up_rounds: 3,
	line_up_max_cm: 4,
	/** Things the jaws must never come down on; descend asks about each first. */
	keep_off: ["computer keyboard"],
	/** How far the fixed jaw's inner face sits left of the object's near side. */
	clearance_cm: 1.5,
	radius_cm: 3.2,
	/**
	 * How far past the object's axis the jaw tips go at a 55° tilt. Grips held with the tips 1.2-6 cm past the bottle's
	 * axis and missed (shoving it out) below that; deep errs safe.
	 */
	depth_cm: 3,
	lift_cm: 6,
	/** Lifting straight up stalls the shoulder past ~28 cm with a load; pull in while lifting. */
	lift_pull_in_cm: 4,
};

/** Overview poses to scan from when the object is not in view. */
export const SCAN: readonly Partial<Spot & { pitch_deg: number }>[] = [
	{ pan_deg: 0, reach_cm: 16, height_cm: 22, pitch_deg: 40 },
	{ pan_deg: -25, reach_cm: 16, height_cm: 22, pitch_deg: 40 },
	{ pan_deg: 25, reach_cm: 16, height_cm: 22, pitch_deg: 40 },
	{ pan_deg: 0, reach_cm: 14, height_cm: 16, pitch_deg: 60 },
	{ pan_deg: -30, reach_cm: 14, height_cm: 16, pitch_deg: 60 },
	{ pan_deg: 30, reach_cm: 14, height_cm: 16, pitch_deg: 60 },
];

type Box = { left: number; top: number; right: number; bottom: number };
/**
 * The same from SAM 3's mask, which hugs the object and leaves its shadow out, so its lowest pixel is the base's near
 * edge. Null when armd has no segmenter, or the mask is missing or cut off by the picture's edge.
 */
async function samAxis(r: Robot, what: string): Promise<{ axis: Spot; box: Box; how: string } | null> {
	const seen = await r.segment(what, { height_cm: GRASP.mid_height_cm });
	if (!seen.found || !seen.whole) return null;
	const steep = (seen.rays.centre?.dir[2] ?? 0) < -GRASP.steep_cos;
	let axis: Spot;
	if (steep && seen.centre) axis = { pan_deg: seen.centre.pan_deg, reach_cm: seen.centre.reach_cm, height_cm: 0 };
	else if (seen.foot) {
		const from = await r.arm();
		const out = seen.foot.reach_cm >= from.reach_cm ? 1 : -1;
		axis = { pan_deg: seen.foot.pan_deg, reach_cm: r1(seen.foot.reach_cm + out * GRASP.radius_cm), height_cm: 0 };
	} else return null;
	if (axis.reach_cm < GRASP.min_reach_cm || axis.reach_cm > GRASP.max_reach_cm) return null;
	return { axis, box: seen.box as Box, how: steep ? "SAM mask centre, from above" : "SAM mask base, from the side" };
}

/**
 * The object's axis from one view. Looking steeply down, the box's centre lies over the axis whatever the object's
 * height; from further off, the box's bottom edge is the object's near side on the table, and the axis one radius on.
 */
async function seeAxis(r: Robot, what: string): Promise<{ axis: Spot; box: Box; how: string } | null> {
	const sam = await samAxis(r, what).catch(() => null);
	if (sam) return sam;
	const seen = await r.locate(what, { height_cm: GRASP.mid_height_cm });
	if (!seen.found) return null;
	const b = seen.box;
	if (Math.min(b.left, b.top, 1 - b.right, 1 - b.bottom) < GRASP.edge_margin) {
		r.log(`find ${what}: box ${JSON.stringify(b)} runs off the picture's edge, so its centre and bottom are not the object's; ignored`);
		return null;
	}
	const steep = (seen.rays.centre?.dir[2] ?? 0) < -GRASP.steep_cos;
	let axis: Spot;
	if (steep && seen.centre) axis = { pan_deg: seen.centre.pan_deg, reach_cm: seen.centre.reach_cm, height_cm: 0 };
	else if (seen.foot) {
		const from = await r.arm();
		const out = seen.foot.reach_cm >= from.reach_cm ? 1 : -1;
		axis = { pan_deg: seen.foot.pan_deg, reach_cm: r1(seen.foot.reach_cm + out * GRASP.radius_cm), height_cm: 0 };
	} else return null;
	if (axis.reach_cm < GRASP.min_reach_cm || axis.reach_cm > GRASP.max_reach_cm) {
		r.log(`find ${what}: box ${JSON.stringify(seen.box)} puts it at reach ${axis.reach_cm} cm, out of the arm's range; ignored`);
		return null;
	}
	return { axis, box: seen.box, how: steep ? "box centre, from above" : "box bottom, from the side" };
}

const apart = (a: Spot, b: Spot) => Math.hypot(a.reach_cm - b.reach_cm, ((a.pan_deg - b.pan_deg) / DEG) * (a.reach_cm + b.reach_cm) / 2);

/**
 * A pose to look at `axis` from: 10 cm nearer the base and above, where the box's bottom edge shows the object's base;
 * an object close to the base is seen steeply from above instead.
 */
export const lookAt = (axis: Spot, nudge_cm = 0) =>
	axis.reach_cm >= 24
		? { pan_deg: axis.pan_deg, reach_cm: r1(axis.reach_cm - 10 + nudge_cm), height_cm: 17, pitch_deg: axis.reach_cm >= 30 ? 35 : 50 }
		: { pan_deg: axis.pan_deg, reach_cm: r1(Math.max(14, axis.reach_cm - 4 + nudge_cm)), height_cm: 16, pitch_deg: 60 };

/**
 * Where an upright object's axis stands. Jev's box can run long past the object's base, so one view is not trusted:
 * look again from a pose aimed at the first answer, and keep answers that agree within `agree_cm`.
 */
export async function find(r: Robot, what: string, o: { agree_cm?: number; views?: number } = {}): Promise<SkillResult & { axis?: Spot }> {
	if ((await r.arm()).height_cm < GRASP.hover_cm) await goTo(r, { height_cm: GRASP.hover_cm + 3 });
	let first = await seeAxis(r, what);
	for (let i = 0; !first && i < SCAN.length; i++) {
		await goTo(r, SCAN[i]!);
		first = await seeAxis(r, what);
	}
	if (!first) return { ok: false, why: `${what} not in view from any scan pose` };
	const seen = [first.axis];
	r.log(`find ${what}: box ${JSON.stringify(first.box)} (${first.how}) -> axis ${JSON.stringify(first.axis)}`);
	for (let v = 1; v < (o.views ?? 4); v++) {
		const last = seen.at(-1)!;
		await goTo(r, lookAt(last, v === 1 ? 0 : -3));
		const next = await seeAxis(r, what);
		if (!next) continue;
		r.log(`find ${what}: box ${JSON.stringify(next.box)} (${next.how}) -> axis ${JSON.stringify(next.axis)}`);
		const match = seen.find((s) => apart(s, next.axis) <= (o.agree_cm ?? 2));
		seen.push(next.axis);
		if (match) {
			const axis = { pan_deg: r1((match.pan_deg + next.axis.pan_deg) / 2), reach_cm: r1((match.reach_cm + next.axis.reach_cm) / 2), height_cm: 0 };
			return { ok: true, why: `two views agree within ${o.agree_cm ?? 2} cm`, axis };
		}
	}
	if (seen.length === 1 && first.how.includes("above")) return { ok: true, why: "one view from above; the others did not see it", axis: first.axis };
	return { ok: false, why: `views disagree: ${JSON.stringify(seen)}` };
}

/** The grasp-point spot that puts the fixed jaw `clearance` left of an upright object's side; the moving jaw closes from the right. */
export function besideLeft(axis: Spot, height_cm: number, o: { radius_cm?: number; clearance_cm?: number; pitch_deg?: number } = {}): Spot {
	const side = (o.radius_cm ?? GRASP.radius_cm) + (o.clearance_cm ?? GRASP.clearance_cm);
	return { pan_deg: r1(axis.pan_deg - (side / axis.reach_cm) * DEG), reach_cm: r1(axis.reach_cm + depthAt(o.pitch_deg ?? GRASP.pitch_deg)), height_cm };
}

/** How far past the object's axis the tips go at a tilt, scaled from GRASP.depth_cm at 55°. */
export const depthAt = (pitch_deg: number) => r1((GRASP.depth_cm * Math.cos((pitch_deg * Math.PI) / 180)) / Math.cos((55 * Math.PI) / 180));

/**
 * Whether the object rides in the jaws: a held object moves with the wrist camera, so its box barely shifts when the
 * arm rises `cm`; one left on the desk shrinks and slides. Held a few cm from the lens, its box often runs off the
 * picture and takes in the jaws, so Jev's eye on the grip backs the box test up. Grip current alone also reads
 * "holding" on a jaw jammed on an edge or on something else.
 */
export async function holds(r: Robot, what: string, cm = 3): Promise<{ held: boolean; why: string }> {
	const a = await r.locate(what);
	if (!a.found) return { held: false, why: `${what} not in view` };
	await goBy(r, { height_cm: cm });
	const b = await r.locate(what);
	if (!b.found) return { held: false, why: `${what} left the view when the arm rose ${cm} cm` };
	const centre = (x: Box) => [(x.left + x.right) / 2, (x.top + x.bottom) / 2] as const;
	const [ax, ay] = centre(a.box);
	const [bx, by] = centre(b.box);
	const moved = Math.hypot(bx - ax, by - ay);
	const grew = (b.box.right - b.box.left) / Math.max(0.01, a.box.right - a.box.left);
	const rode = moved <= GRASP.held_max_shift && grew > 0.85;
	const seen = await r.check(`Is the ${what} held between the gripper's two jaws?`);
	const held = rode || seen >= GRASP.held_seen;
	return { held, why: `box moved ${moved.toFixed(3)} and scaled ${grew.toFixed(2)}x when the arm rose ${cm} cm; held between the jaws by eye p ${seen}` };
}

/** The spot `cm` to the arm's left (negative: right) of `from` along a straight line, at the same reach. */
export function sideways(from: Spot, cm: number): Spot {
	return { ...from, pan_deg: r1(from.pan_deg - 2 * Math.asin(cm / 2 / from.reach_cm) * DEG) };
}

const spotOf = (a: { pan_deg: number; reach_cm: number; height_cm: number }): Spot => ({ pan_deg: a.pan_deg, reach_cm: a.reach_cm, height_cm: a.height_cm });

async function carryTo(r: Robot, s: Scratch, cm: number): Promise<SkillResult> {
	const held = s.held!;
	const to = sideways(held.grasp, cm);
	s.place = to;
	const now = await r.arm();
	await goBy(r, { pan_deg: to.pan_deg - now.pan_deg });
	const at = await goTo(r, { reach_cm: to.reach_cm });
	return { ok: true, why: `carried over the spot ${Math.abs(cm)} cm ${cm >= 0 ? "left" : "right"} of where it was picked up; ${at.result}` };
}


/**
 * Jev's eye for the last centimetres: move the open jaws, and take the object to be where the move says it is.
 * Sideways only from above: at the object's height a jaw would shove it. In and out are fine at any height.
 */
function nudge(name: string, says: string, dir: { side?: 1 | -1; out?: 1 | -1 }): Skill {
	return {
		name,
		says,
		amount: { says: "How far should the jaws move?", unit: "cm", levels: [0.5, 1, 2, 3, 5] },
		ready: (f) => f.object_located && !f.object_held && (Boolean(dir.out) || f.jaw_tip_height_cm >= GRASP.clear_cm),
		run: async (r, a, s) => {
			const cm = a.amount ?? 1;
			const now = await r.arm();
			const pan = dir.side ? (-dir.side * cm / now.reach_cm) * DEG : 0;
			const out = dir.out ? dir.out * cm : 0;
			const m = await goBy(r, { ...(pan ? { pan_deg: r1(pan) } : {}), ...(out ? { reach_cm: out } : {}) });
			if (s.target) s.target = { ...s.target, pan_deg: r1(s.target.pan_deg + pan), reach_cm: r1(s.target.reach_cm + out) };
			return { ok: m.result === "moved", why: `jaws moved ${cm} cm ${name.replace("nudge_", "")}: ${m.result}` };
		},
	};
}


/** Moves the open jaws by a correction: sideways moves rise above the object first so the jaws don't shove it. */
async function shift(r: Robot, s: Scratch, right_cm: number, out_cm: number): Promise<void> {
	const now = await r.arm();
	const pan = r1((right_cm / now.reach_cm) * DEG);
	if (Math.abs(right_cm) >= 0.3) {
		const low = now.height_cm < GRASP.clear_cm;
		if (low) await goTo(r, { height_cm: GRASP.clear_cm });
		await goBy(r, { pan_deg: pan, ...(out_cm ? { reach_cm: out_cm } : {}) });
		if (low) await goTo(r, { height_cm: now.height_cm });
	} else if (out_cm) await goBy(r, { reach_cm: out_cm });
	if (s.target) s.target = { ...s.target, pan_deg: r1(s.target.pan_deg + (Math.abs(right_cm) >= 0.3 ? pan : 0)), reach_cm: r1(s.target.reach_cm + out_cm) };
}

/** Claude lines the open jaws up on the object, a few rounds of look and move; null when there is no Claude to ask. */
async function lineUp(r: Robot, a: SkillArgs, s: Scratch): Promise<SkillResult | null> {
	if (!a.oversee) return null;
	const moves: string[] = [];
	for (let round = 0; round < GRASP.line_up_rounds; round++) {
		const v = await a.oversee("aligned");
		if (!v) return null;
		if (v.agree) return { ok: true, why: `Claude sees the object in the jaw opening${moves.length ? ` after ${moves.join(", ")}` : ""}` };
		const m = v.move;
		if (!m || (Math.abs(m.right_cm) < 0.3 && Math.abs(m.out_cm) < 0.3)) return { ok: false, why: `Claude does not see the object in the opening and gave no move: ${v.why}` };
		const right = Math.max(-GRASP.line_up_max_cm, Math.min(GRASP.line_up_max_cm, m.right_cm));
		const out = Math.max(-GRASP.line_up_max_cm, Math.min(GRASP.line_up_max_cm, m.out_cm));
		await shift(r, s, right, out);
		moves.push(`${r1(right)} cm right, ${r1(out)} cm out`);
	}
	return { ok: false, why: `not lined up after ${GRASP.line_up_rounds} rounds (${moves.join("; ")})` };
}

/** Small steps; the chooser picks one at a time from the picture and the facts, and each does its own careful motion. */
export const SKILLS: readonly Skill[] = [
	{
		name: "find_object",
		says: "Look for the object and work out where it stands on the desk.",
		ready: (f) => !f.object_held,
		checkpoint: { kind: "found", refute: (s) => void (s.target = undefined) },
		run: async (r, a, s) => {
			const found = await find(r, a.what);
			if (found.axis) s.target = found.axis;
			s.set_down = false;
			return found;
		},
	},
	{
		name: "approach",
		says: "Open the jaws and move them above the object, pointing down as steeply as the arm can, just to its left, ready to come straight down around it.",
		ready: (f) => f.object_located && !f.object_held,
		run: async (r, a, s) => {
			await letGo(r, s);
			const tried: string[] = [];
			for (const pitch_deg of GRASP.approach_pitches)
				for (const hover of [GRASP.clear_cm + 1, GRASP.clear_cm]) {
					try {
						await r.moveTo({ ...besideLeft(s.target!, hover, { pitch_deg }), pitch_deg });
						s.pitch_deg = pitch_deg;
						return { ok: true, why: `jaws open ${hover} cm up, above and left of the object at ${JSON.stringify(s.target)}, tilted ${pitch_deg}°` };
					} catch (e) {
						if (!String((e as Error).message).startsWith("refused")) throw e;
						tried.push(`${pitch_deg}° at ${hover} cm`);
					}
				}
			return { ok: false, why: `the arm cannot reach above the object at any tilt tried (${tried.join(", ")}); it may be too far out` };
		},
	},
	nudge("nudge_left", "Move the jaws a little toward image left (the arm's left), to line the object up between them.", { side: 1 }),
	nudge("nudge_right", "Move the jaws a little toward image right (the arm's right), to line the object up between them.", { side: -1 }),
	nudge("nudge_out", "Move the jaws a little away from the base (toward image up).", { out: 1 }),
	nudge("nudge_in", "Move the jaws a little toward the base (toward image down).", { out: -1 }),
	{
		name: "descend",
		says: "Lower the open jaws straight down around the object, when the object is just right of the fixed jaw with room in between.",
		amount: { says: "How high above the desk should the jaw tips stop? Above the object's middle keeps it upright when lifted.", unit: "cm", levels: [3, 4, 5, 6, 8] },
		ready: (f) => f.jaws_beside_object && !f.object_held,
		run: async (r, a, s) => {
			if ((await r.arm()).grip !== "open") await letGo(r, s);
			for (const thing of GRASP.keep_off) {
				const p = await r.check(`Is a ${thing} directly below the gripper's jaws, where they would land if they moved straight down?`);
				if (p > 0.5) return { ok: false, why: `refused to go down: a ${thing} looks to be under the jaws (p ${p})` };
			}
			const down = await goTo(r, { height_cm: a.amount ?? GRASP.grip_cm });
			const said = `jaw tips at ${down.arm.height_cm} cm: ${down.result}`;
			const lined = await lineUp(r, a, s);
			return lined ? { ok: lined.ok, why: `${said}; ${lined.why}` } : { ok: down.result === "moved", why: said };
		},
	},
	{
		name: "line_up",
		says: "Ask Claude to look and move the open jaws a little, until the object sits in the jaw opening ready to close on.",
		ready: (f) => f.object_located && !f.object_held && f.gripper === "open" && f.jaw_tip_height_cm < GRASP.grip_zone_cm,
		run: async (r, a, s) => (await lineUp(r, a, s)) ?? { ok: false, why: "no Claude to ask in this run" },
	},
	{
		name: "close",
		says: "Close the jaws on the object.",
		ready: (f) => f.gripper === "open" && !f.object_held && f.jaw_tip_height_cm < GRASP.grip_zone_cm,
		run: async (r, _a, s) => {
			const g = await r.grip("close");
			s.closed_at = spotOf(g.arm);
			if (g.grip === "empty") {
				await r.grip("open", GRASP.open_pct);
				s.target = undefined;
				return { ok: false, why: "the jaws closed on nothing, so the object is not between them; a missed close usually pushes it away, so find it again" };
			}
			const why = { holding: "the jaws stopped part way: something is between them", empty: "the jaws closed on nothing", jammed: "the jaws stopped almost at once", open: "the jaws are open", dropped: "it slipped out", unknown: "unknown" }[g.grip as string] ?? g.grip;
			return { ok: g.grip === "holding", why };
		},
	},
	{
		name: "lift",
		says: "The jaws have closed on something: lift it off the desk and check by sight that it is the object riding in the jaws.",
		ready: (f) => f.gripper === "holding" && !f.object_held,
		checkpoint: {
			kind: "gripped",
			refute: (s) => {
				s.held = undefined;
				s.bad_grip = true;
			},
		},
		run: async (r, a, s) => {
			const before = await r.arm();
			const up = await goBy(r, { reach_cm: -GRASP.lift_pull_in_cm, height_cm: GRASP.lift_cm });
			if (up.arm.grip !== "holding") return { ok: false, why: `the grip reads ${up.arm.grip} after lifting` };
			const rides = await holds(r, a.what);
			let held = rides.held;
			let why = rides.why;
			let claude = false;
			if (!held && a.oversee) {
				const v = await a.oversee("gripped");
				if (v) {
					claude = true;
					held = v.agree;
					why = `${why}; Claude ${v.agree ? "sees it held" : "agrees it is not held"}: ${v.why}`;
				}
			}
			if (held) s.held = { what: a.what, grasp: s.closed_at ?? spotOf(before) };
			else s.bad_grip = true;
			return { ok: held, why: held ? `the object rides in the jaws (${why})` : `the jaws hold something, but not the object (${why})`, ...(claude ? { claude_checked: true } : {}) };
		},
	},
	{
		name: "move_left",
		says: "Carry the held object to the arm's left, at its carrying height.",
		amount: { says: "How far left of where it was picked up should it go?", unit: "cm", levels: [5, 8, 10, 15, 20] },
		ready: (f) => f.object_held && !f.at_set_down_height,
		run: (r, a, s) => carryTo(r, s, a.amount ?? 10),
	},
	{
		name: "move_right",
		says: "Carry the held object to the arm's right, at its carrying height.",
		amount: { says: "How far right of where it was picked up should it go?", unit: "cm", levels: [5, 8, 10, 15, 20] },
		ready: (f) => f.object_held && !f.at_set_down_height,
		run: (r, a, s) => carryTo(r, s, -(a.amount ?? 10)),
	},
	{
		name: "lower",
		says: "Lower the held object straight down until it stands on the desk, here.",
		ready: (f) => f.object_held && !f.at_set_down_height,
		run: async (r, _a, s) => {
			const at = s.held!.grasp.height_cm;
			await goTo(r, { height_cm: at + 2.5 });
			const down = await goTo(r, { height_cm: at + 0.3 });
			return { ok: true, why: `jaw tips at ${down.arm.height_cm} cm, where they gripped it: ${down.result}` };
		},
	},
	{
		name: "open",
		says: "Open the jaws and let go: to set the held object down on the desk, or to let go of a grip that missed it.",
		ready: (f) => f.gripper !== "open" && (f.at_set_down_height || (!f.object_held && !freshGrip(f))),
		run: async (r, _a, s) => {
			if (s.held) {
				await r.grip("open", GRASP.open_pct);
				s.held = undefined;
				s.set_down = true;
				return { ok: true, why: "let go of the object on the desk" };
			}
			return { ok: true, why: await letGo(r, s) };
		},
	},
	{
		name: "raise",
		says: "Raise the open jaws straight up above the object, to line up again from above.",
		ready: (f) => !f.object_held && f.jaw_tip_height_cm < GRASP.clear_cm,
		run: async (r) => {
			const up = await goTo(r, { height_cm: GRASP.hover_cm });
			return { ok: up.result === "moved", why: `jaw tips at ${up.arm.height_cm} cm: ${up.result}` };
		},
	},
	{
		name: "retreat",
		says: "Raise the gripper straight up, clear of the object, and back off.",
		run: async (r, _a, s) => {
			await goTo(r, { height_cm: Math.max(GRASP.hover_cm, (await r.arm()).height_cm) });
			if (!s.held) await goBy(r, { reach_cm: -5 });
			return { ok: true, why: "raised clear" };
		},
	},
];

export function skillNamed(name: string): Skill {
	const s = SKILLS.find((k) => k.name === name);
	if (!s) throw new Error(`no skill ${name}; skills: ${SKILLS.map((k) => k.name).join(", ")}`);
	return s;
}

export function bindSkills(r: Robot, scratch: Scratch = {}) {
	return {
		skill: (name: string, args: SkillArgs) => skillNamed(name).run(r, args, scratch),
		find: (what: string) => find(r, what),
	};
}
