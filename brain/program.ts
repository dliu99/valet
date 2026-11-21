import { createContext, Script } from "node:vm";
import { ArmError, CAMS, type ArmClient, type ArmState, type Frame, type Frames, type Motion, type Ray } from "./arm.ts";
import { yesProbability, type DecisionRequest } from "./decision.ts";
import type { RecordedModel, Recorder } from "./record.ts";
import { bindSkills, type Scratch } from "./skills.ts";
import { runTask } from "./stage.ts";
import { WRIST_CONVENTION } from "./task.ts";

export interface ProgramDeps {
	arm: ArmClient;
	model: RecordedModel;
	recorder: Recorder;
	view: string;
}

export interface ProgramResult {
	ok: boolean;
	error: string | null;
	log: string[];
	arm: ArmState | null;
	frames: Frames;
	ms: number;
}

export const LOCATE_MIN_CONFIDENCE = 0.3;
/** The wrist camera's picture lags the arm: a frame taken sooner than this after a move can still show the old view. */
export const SETTLE_MS = 700;
export const settle = () => new Promise((resolve) => setTimeout(resolve, SETTLE_MS));
export const OPEN_GAP_CM = 7.5;
export const JAW_CLEARANCE_CM = 0.5;

export function jawsAt(t: { pan_deg: number; reach_cm: number; height_cm: number }, o: { width_cm?: number; gap_cm?: number; pitch_deg: number; depth_cm?: number }) {
	const r = (v: number) => Math.round(v * 10) / 10;
	const side = Math.max((o.gap_cm ?? OPEN_GAP_CM) / 2, (o.width_cm ?? 0) / 2 + JAW_CLEARANCE_CM);
	const depth = o.depth_cm ?? 2;
	const pitch = (o.pitch_deg * Math.PI) / 180;
	return {
		pan_deg: r(t.pan_deg - (side / t.reach_cm) * (180 / Math.PI)),
		reach_cm: r(t.reach_cm + depth * Math.cos(pitch)),
		height_cm: r(t.height_cm - depth * Math.sin(pitch)),
		pitch_deg: o.pitch_deg,
	};
}

interface Pose {
	pan_deg?: number;
	reach_cm?: number;
	height_cm?: number;
	pitch_deg?: number;
	roll_deg?: number;
}

export function armSummary(arm: ArmState) {
	const r = (v: number) => Math.round(v * 10) / 10;
	return {
		pan_deg: r(arm.pan_deg),
		reach_cm: r(arm.reach_m * 100),
		height_cm: r(arm.height_m * 100),
		pitch_deg: r(arm.pitch_deg),
		roll_deg: r(arm.roll_deg),
		gripper_pct: r(arm.gripper_pct),
		grip: arm.grip,
	};
}

export type Robot = ReturnType<typeof makeRobot>;

/** The arm API programs and skills use: distances in cm, angles in degrees. */
export function makeRobot(id: string, deps: ProgramDeps, stop: AbortSignal, log: string[]) {
	let calls = 0;
	let movedAt = 0;
	const settled = async () => {
		const wait = movedAt + SETTLE_MS - performance.now();
		if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
	};
	const live = () => {
		if (stop.aborted) throw stop.reason;
		calls++;
	};
	const wrist = async (): Promise<Frame> => {
		await settled();
		const look = await deps.arm.frames(["wrist"], deps.model.caps.maxSide);
		if (!look.frames.wrist) throw new Error(`no wrist frame (${look.missing.wrist ?? "missing"})`);
		return look.frames.wrist;
	};
	const ask = async (questions: DecisionRequest["questions"]) => {
		const [arm, frame] = await Promise.all([deps.arm.state(), wrist()]);
		const request: DecisionRequest = { state: { camera: `${deps.view}. ${WRIST_CONVENTION}`, arm: armSummary(arm) }, images: { wrist: frame }, questions };
		return { frame, answers: await deps.model.decide(request, { role: "ask", task: id, tick: calls, facts: { arm, recent: [] } }) };
	};
	const motion = async (m: Motion) => {
		const t0 = performance.now();
		try {
			const r = await deps.arm.act(m).finally(() => void (movedAt = performance.now()));
			await deps.recorder.motion({ task: id, tick: calls, motion: m, result: r as never, error: null, ms: Math.round(performance.now() - t0) });
			if (!r.ok && !r.stopped) throw new Error(`refused: ${r.refused}`);
			return { result: r.ok ? "moved" : `stopped: ${r.stopped}`, arm: armSummary(r.after) };
		} catch (e) {
			const error = e instanceof ArmError ? `${e.kind}: ${e.message}` : (e as Error).message;
			await deps.recorder.motion({ task: id, tick: calls, motion: m, result: null, error, ms: Math.round(performance.now() - t0) });
			if (e instanceof ArmError && e.refusal) throw new Error(`refused: ${error}`);
			throw e;
		}
	};
	const metres = (p: Pose) => ({
		...(p.pan_deg !== undefined ? { pan_deg: p.pan_deg } : {}),
		...(p.reach_cm !== undefined ? { reach_m: p.reach_cm / 100 } : {}),
		...(p.height_cm !== undefined ? { height_m: p.height_cm / 100 } : {}),
		...(p.pitch_deg !== undefined ? { pitch_deg: p.pitch_deg } : {}),
		...(p.roll_deg !== undefined ? { roll_deg: p.roll_deg } : {}),
	});
	const cm = (m: number) => Math.round(m * 1000) / 10;
	const onPlane = async (u: number, v: number, aspect: number, heightCm: number) => {
		const hit = await deps.arm.locate(u, v, aspect, heightCm / 100);
		return { at: hit.hit ? { pan_deg: hit.pan_deg, reach_cm: cm(hit.reach_m), height_cm: heightCm } : null, ray: hit.ray };
	};

	return {
		log: (...args: unknown[]) => void log.push(args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" ")),
		sleep: async (ms: number) => {
			live();
			await new Promise((resolve) => setTimeout(resolve, Math.min(ms, 10_000)));
			live();
		},
		arm: async () => (live(), armSummary(await deps.arm.state())),
		moveTo: async (p: Pose) => (live(), motion({ kind: "to", ...metres(p) })),
		move: async (p: Pose) => (live(), motion({ kind: "delta", ...metres(p) })),
		pose: async (name: string) => (live(), motion({ kind: "pose", name })),
		grip: async (to: "open" | "close", pct?: number) => {
			live();
			const r = await motion({ kind: "grip", to, ...(to === "open" && pct !== undefined ? { pct } : {}) });
			return { ...r, grip: r.arm.grip };
		},
		check: async (question: string) => {
			live();
			const { answers } = await ask({ check: { type: "bool", instructions: question, criteria: { true: "Yes.", false: "No, or the image does not show it." } } });
			return Math.round((yesProbability(answers.check) ?? 0) * 100) / 100;
		},
		locate: async (what: string, opts: { height_cm?: number } = {}) => {
			live();
			const phrasings = [`the ${what}`, `the ${what} itself, not its shadow on the desk`, `the whole ${what}, the object only`];
			const { frame, answers } = await ask(Object.fromEntries(phrasings.map((t, i) => [`box:${i}`, { type: "box" as const, instructions: `${t} in \`images.wrist\`` }])));
			const boxes = Object.values(answers).flatMap((x) => (x?.type === "box" && x.confidence >= LOCATE_MIN_CONFIDENCE ? [x] : []));
			if (boxes.length * 2 < phrasings.length) return { found: false as const, confidence: Math.max(0, ...Object.values(answers).map((x) => (x?.type === "box" ? x.confidence : 0))) };
			const median = (vs: number[]) => [...vs].sort((x, y) => x - y)[Math.floor((vs.length - 1) / 2)]!;
			const edge = (k: "left" | "top" | "right" | "bottom") => median(boxes.map((x) => x.box[k]));
			const a = { box: { left: edge("left"), top: edge("top"), right: edge("right"), bottom: edge("bottom") }, confidence: median(boxes.map((x) => x.confidence)) };
			const aspect = frame.width / frame.height;
			const cx = (a.box.left + a.box.right) / 2;
			const cy = (a.box.top + a.box.bottom) / 2;
			const box = Object.fromEntries(Object.entries(a.box).map(([k, v]) => [k, Math.round(v * 1000) / 1000])) as unknown as typeof a.box;
			const h = opts.height_cm ?? 0;
			const [centre, foot, top, left, right] = await Promise.all([onPlane(cx, cy, aspect, h), onPlane(cx, a.box.bottom, aspect, 0), onPlane(cx, a.box.top, aspect, 0), onPlane(a.box.left, cy, aspect, h), onPlane(a.box.right, cy, aspect, h)]);
			return { found: true as const, confidence: Math.round(a.confidence * 100) / 100, box, centre: centre.at, foot: foot.at, left: left.at, right: right.at, ray: centre.ray, rays: { centre: centre.ray, top: top.ray, bottom: foot.ray } };
		},
		/** SAM 3's mask of `what` in the wrist frame: its stats in u, v of the upright frame, and where its centroid
		 * (on a plane height_cm above the table) and its bottom-centre (on the table) lie. Not found below score 0.5. */
		segment: async (what: string, opts: { height_cm?: number } = {}) => {
			live();
			await settled();
			const s = await deps.arm.segment([what]);
			const r = s.results[what];
			const frame = s.frame.sha256;
			if (!r?.found) return { found: false as const, score: r?.score ?? 0, frame };
			const aspect = s.frame.width / s.frame.height;
			const [centre, foot] = await Promise.all([onPlane(r.centroid.u, r.centroid.v, aspect, opts.height_cm ?? 0), onPlane(r.bottom.u, r.bottom.v, aspect, 0)]);
			return { ...r, centre: centre.at, foot: foot.at, ray: centre.ray, rays: { centre: centre.ray, bottom: foot.ray }, frame, ms: s.ms };
		},
		jawsAt: (t: { pan_deg: number; reach_cm: number; height_cm: number }, o: { width_cm?: number; gap_cm?: number; pitch_deg: number; depth_cm?: number }) => jawsAt(t, o),
		triangulate: async (a: { ray?: Ray }, b: { ray?: Ray }) => {
			live();
			if (!a?.ray || !b?.ray) return { ok: false as const, why: "one of the two locate() results found nothing" };
			const t = await deps.arm.triangulate(a.ray, b.ray);
			return t.ok ? { ok: true as const, pan_deg: t.pan_deg, reach_cm: cm(t.reach_m), height_cm: cm(t.height_m), gap_cm: cm(t.gap_m), angle_deg: t.angle_deg } : t;
		},
		jev: async (goal: string, opts: { notes?: string; seconds?: number } = {}) => {
			live();
			const report = await runTask({ id: `${id}.jev${calls}`, goal, seconds: opts.seconds ?? 30, ...(opts.notes ? { notes: opts.notes } : {}) }, deps, stop);
			return { exit: report.exit, reason: report.reason, steps: report.steps.map((s) => `${s.action}${s.step ? ` ${s.step}` : ""} (p ${s.p.toFixed(2)}): ${s.did}`) };
		},
	};
}

export async function runProgram(id: string, code: string, deps: ProgramDeps, signal: AbortSignal, seconds: number, scratch: Scratch = {}): Promise<ProgramResult> {
	const started = performance.now();
	const log: string[] = [];
	const stop = new AbortController();
	const onAbort = () => stop.abort(new Error("stopped from outside"));
	signal.addEventListener("abort", onAbort, { once: true });
	const deadline = setTimeout(() => stop.abort(new Error(`the program ran past its ${seconds} s`)), seconds * 1000);
	const robot = makeRobot(id, deps, stop.signal, log);
	const api = { ...robot, ...bindSkills(robot, scratch) };

	let error: string | null = null;
	try {
		await deps.arm.hold();
		const script = new Script(`(async () => {\n${code}\n})()`, { filename: `program-${id}.js` });
		const run = script.runInContext(createContext(Object.freeze({ ...api })), { timeout: 1000 }) as Promise<unknown>;
		const aborted = new Promise<never>((_, reject) => stop.signal.addEventListener("abort", () => reject(stop.signal.reason), { once: true }));
		aborted.catch(() => undefined);
		await Promise.race([run, aborted]);
	} catch (e) {
		error = (e as Error)?.message ?? String(e);
		if (stop.signal.aborted) await deps.arm.hold().catch(() => undefined);
	} finally {
		clearTimeout(deadline);
		signal.removeEventListener("abort", onAbort);
		stop.abort(new Error("the program ended"));
	}
	const [arm, look] = await Promise.all([deps.arm.state().catch(() => null), deps.arm.frames(CAMS).catch(() => ({ frames: {} as Frames }))]);
	return { ok: error === null, error, log, arm, frames: look.frames, ms: Math.round(performance.now() - started) };
}
