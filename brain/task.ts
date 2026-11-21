import type { ArmState, Frame, Motion } from "./arm.ts";
import { topChoice, type Answers, type DecisionRequest, type JsonObject, type Question } from "./decision.ts";

export interface Task {
	id: string;
	goal: string;
	notes?: string;
	seconds: number;
}

export type Move = "pan_left" | "pan_right" | "reach_out" | "reach_in" | "raise" | "lower" | "pitch_up" | "pitch_down" | "roll_cw" | "roll_ccw";
export type Action = Move | "open" | "close" | "done" | "stuck";

export const OPTIONS: Record<Action, string> = {
	pan_left: "Turn the base so the gripper moves toward image left (the arm's left).",
	pan_right: "Turn the base so the gripper moves toward image right (the arm's right).",
	reach_out: "Move the gripper straight away from the base (toward image up).",
	reach_in: "Move the gripper straight back toward the base (toward image down).",
	raise: "Move the gripper straight up.",
	lower: "Move the gripper straight down.",
	pitch_up: "Tilt the gripper up, toward horizontal.",
	pitch_down: "Tilt the gripper down, toward pointing straight down.",
	roll_cw: "Roll the wrist clockwise.",
	roll_ccw: "Roll the wrist counter-clockwise.",
	open: "Open the gripper.",
	close: "Close the gripper until it stops on something, then squeeze a little.",
	done: "`goal` is accomplished.",
	stuck: "`goal` cannot be accomplished from here; hand back to the planner.",
};

export const ACTIONS = Object.keys(OPTIONS) as Action[];

export interface Level {
	value: number;
	label: string;
}

const levels = (values: number[], say: (v: number) => string): Level[] => values.map((value) => ({ value, label: say(value) }));
const TRAVEL_CM = levels([0.5, 1, 2, 4, 8], (v) => `About ${v} cm.`);
const TILT_DEG = levels([2, 5, 10, 20, 30], (v) => `About ${v} degrees.`);
const ROLL_DEG = levels([5, 10, 20, 45, 90], (v) => `About ${v} degrees.`);

type Axis = "pan_deg" | "reach_m" | "height_m" | "pitch_deg" | "roll_deg";

export const MOVES: Record<Move, { axis: Axis; sign: 1 | -1; levels: readonly Level[] }> = {
	pan_left: { axis: "pan_deg", sign: -1, levels: TRAVEL_CM },
	pan_right: { axis: "pan_deg", sign: 1, levels: TRAVEL_CM },
	reach_out: { axis: "reach_m", sign: 1, levels: TRAVEL_CM },
	reach_in: { axis: "reach_m", sign: -1, levels: TRAVEL_CM },
	raise: { axis: "height_m", sign: 1, levels: TRAVEL_CM },
	lower: { axis: "height_m", sign: -1, levels: TRAVEL_CM },
	pitch_down: { axis: "pitch_deg", sign: 1, levels: TILT_DEG },
	pitch_up: { axis: "pitch_deg", sign: -1, levels: TILT_DEG },
	roll_cw: { axis: "roll_deg", sign: 1, levels: ROLL_DEG },
	roll_ccw: { axis: "roll_deg", sign: -1, levels: ROLL_DEG },
};

const isMove = (a: Action): a is Move => a in MOVES;
const MIN_PAN_RADIUS_M = 0.1;
export const RECENT = 8;
export const SURE = 0.35;
export const UNSURE_HANDBACK = 2;

export function farId(move: Move): string {
	return `far_${move}`;
}

export function levelValue(levels: readonly Level[], score: number): number {
	const top = levels.length - 1;
	const s = Math.min(top, Math.max(0, score));
	const i = Math.min(top - 1, Math.floor(s));
	return levels[i]!.value + (s - i) * (levels[i + 1]!.value - levels[i]!.value);
}

export function travelToPanDeg(cm: number, reachM: number): number {
	return ((cm / 100 / Math.max(reachM, MIN_PAN_RADIUS_M)) * 180) / Math.PI;
}

export const WRIST_CONVENTION =
	"The wrist image is upright: image up is away from the robot's base, image right is the arm's right. With the gripper open only the fixed jaw is in view; the opening is to its right.";

const GRIP_MEANING: Record<ArmState["grip"], string> = {
	open: "open",
	holding: "holding: told to close, it stopped part way, so something is between the jaws",
	jammed: "jammed: told to close, it stopped wide open",
	empty: "empty: told to close, it closed fully on nothing",
	unknown: "unknown",
};

export interface Step {
	tick: number;
	action: Action;
	p: number;
	step: string | null;
	did: string;
}

export interface Scene {
	arm: ArmState;
	frame: Frame;
	recent: readonly Step[];
	view: string;
}

export function validate(task: Task): string[] {
	const errors: string[] = [];
	if (!task || typeof task.goal !== "string" || !task.goal.trim()) errors.push("goal: what Jev should achieve");
	if (task?.notes !== undefined && typeof task.notes !== "string") errors.push("notes: text, or leave it out");
	if (!(task?.seconds > 0)) errors.push("seconds: how long Jev may work before handing back");
	return errors;
}

/** The arm's pose and what its gripper sensed, as Jev reads them in `state`. */
export function armFacts(arm: ArmState): JsonObject {
	return {
		arm: {
			jaw_tip_height_above_table_cm: round(arm.height_m * 100, 1),
			reach_from_base_cm: round(arm.reach_m * 100, 1),
			pan_deg: round(arm.pan_deg, 1),
			pitch_deg: round(arm.pitch_deg, 1),
			wrist_roll_deg: round(arm.roll_deg, 1),
			gripper_open_pct: round(arm.gripper_pct, 0),
		},
		gripper: GRIP_MEANING[arm.grip] ?? arm.grip,
	};
}

export function compile(task: Task, scene: Scene): DecisionRequest {
	const state: JsonObject = {
		goal: task.goal,
		...(task.notes ? { notes: task.notes } : {}),
		...armFacts(scene.arm),
		recent_actions: scene.recent.slice(-RECENT).map((s): JsonObject => ({ action: s.action, ...(s.step ? { step: s.step } : {}), result: s.did })),
		camera: `${scene.view}. ${WRIST_CONVENTION}`,
	};
	const facts = ["`arm`", "`gripper`", "`recent_actions`", ...(task.notes ? ["`notes`"] : [])].join(", ");
	const questions: Record<string, Question> = {
		act: {
			type: "choice",
			instructions: `Which one action should the robot arm take next toward \`goal\`? Judge from \`images.wrist\` together with ${facts}.`,
			criteria: { ...OPTIONS },
		},
	};
	for (const move of Object.keys(MOVES) as Move[])
		questions[farId(move)] = {
			type: "score",
			instructions: `If the next action is "${OPTIONS[move]}", how far should it go toward \`goal\`?`,
			criteria: MOVES[move].levels.map((l) => l.label),
		};
	return { state, images: { wrist: scene.frame }, questions };
}

export interface Reading {
	action: Action;
	p: number;
}

export function read(answers: Answers): Reading {
	const top = topChoice(answers.act);
	if (!top || !(top.choice in OPTIONS)) throw new Error(`the act answer named no action: ${JSON.stringify(answers.act)}`);
	return { action: top.choice as Action, p: top.p };
}

export interface MotionPlan {
	motion: Motion | null;
	step: string | null;
}

export function toMotion(action: Action, answers: Answers, arm: ArmState): MotionPlan {
	if (action === "open" || action === "close") return { motion: { kind: "grip", to: action === "open" ? "open" : "close" }, step: null };
	if (!isMove(action)) return { motion: null, step: null };
	const move = MOVES[action];
	const far = answers[farId(action)];
	const value = levelValue(move.levels, far?.type === "score" ? far.score : 0);
	const cm = move.levels === TRAVEL_CM;
	const amount = move.axis === "pan_deg" ? travelToPanDeg(value, arm.reach_m) : move.axis.endsWith("_m") ? value / 100 : value;
	return { motion: { kind: "delta", [move.axis]: round(move.sign * amount, move.axis.endsWith("_m") ? 4 : 2) }, step: `${value.toFixed(1)} ${cm ? "cm" : "°"}` };
}

function round(v: number, digits: number): number {
	const f = 10 ** digits;
	return Math.round(v * f) / f;
}
