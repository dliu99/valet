// Its kinematics are a stand-in: the pan axis sits 3.9 cm ahead of the base origin and the grasp point 7.7 mm left
// of the radial line, as on the SO-101 at roll -90°.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import type { ArmState, Cam, Frame, MotionResult } from "../arm.ts";

export interface FakeTruth {
	box: { x: number; y: number; z: number; w: number; d: number; h: number };
	held: boolean;
	tip: { x: number; y: number; z: number };
}

const FRAME = readFileSync(new URL("./armd-double.jpg", import.meta.url));
const FRAME_SHA = createHash("sha256").update(FRAME).digest("hex");
const RAD = Math.PI / 180;
const PAN_AXIS = { x: 0.0388, y: 0 };
const LATERAL_M = 0.0077;
const OPEN_PCT = 60;
const JAW_GAP_AT_FULL_OPEN_M = 0.1;
const JAW_REACH_MARGIN_M = 0.015;
const MAX_COMMAND_DEG = 30;
const TABLE_CLEARANCE_M = 0.01;
const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]"]);

interface Pose {
	pan: number;
	reach: number;
	height: number;
	pitch: number;
	roll: number;
}

export interface BoxSpec {
	x: number;
	y: number;
	w: number;
	d: number;
	h: number;
}

export interface DoubleOptions {
	allowMotion?: boolean;
	box?: BoxSpec;
	cams?: Cam[];
	maxReachM?: number;
}

export interface ArmdDouble {
	url: string;
	calls: string[];
	truth(): FakeTruth;
	fail(detail: string): void;
	close(): Promise<void>;
}

export const DEFAULT_BOX: BoxSpec = { x: 0.279, y: 0, w: 0.04, d: 0.04, h: 0.05 };
const START: Pose = { pan: 0, reach: 0.2, height: 0.16, pitch: 60, roll: -90 };

class Refusal extends Error {
	readonly status: number;
	readonly kind: string;
	constructor(status: number, kind: string, message: string) {
		super(message);
		this.status = status;
		this.kind = kind;
	}
}

export async function startArmdDouble(opts: DoubleOptions = {}): Promise<ArmdDouble> {
	const cams = opts.cams ?? ["wrist", "side"];
	const maxReach = opts.maxReachM ?? 0.32;
	let pose: Pose = { ...START };
	let gripper = OPEN_PCT;
	let grip: ArmState["grip"] = "unknown";
	let engaged = false;
	let box = { ...(opts.box ?? DEFAULT_BOX), z: (opts.box ?? DEFAULT_BOX).h / 2 };
	let held: { right: number; ahead: number; up: number } | null = null;
	let failure: string | null = null;
	const poses: Record<string, Pose> = {};
	const calls: string[] = [];

	const fwd = (pan: number) => ({ x: Math.cos(pan * RAD), y: -Math.sin(pan * RAD) });
	const left = (pan: number) => ({ x: Math.sin(pan * RAD), y: Math.cos(pan * RAD) });
	const tip = (p: Pose) => ({
		x: PAN_AXIS.x + p.reach * fwd(p.pan).x + LATERAL_M * left(p.pan).x,
		y: PAN_AXIS.y + p.reach * fwd(p.pan).y + LATERAL_M * left(p.pan).y,
		z: p.height,
	});
	const joints = (p: Pose) => ({
		shoulder_pan: round(p.pan),
		shoulder_lift: round((p.reach - START.reach) * 300 - 27.2),
		elbow_flex: round((START.height - p.height) * 300 - 1),
		wrist_flex: round(p.pitch + 28.2),
		wrist_roll: round(p.roll),
	});
	const state = (): ArmState => ({
		t: Date.now() / 1000,
		kind: "fake",
		allow_motion: opts.allowMotion ?? true,
		joints: { ...joints(pose), gripper },
		tip: tip(pose),
		pan_deg: pose.pan,
		reach_m: pose.reach,
		height_m: pose.height,
		pitch_deg: pose.pitch,
		roll_deg: pose.roll,
		gripper_pct: gripper,
		grip,
		amps: { shoulder_pan: 0.1, shoulder_lift: held ? 0.25 : 0.1, elbow_flex: held ? 0.25 : 0.1, wrist_flex: 0.1, wrist_roll: 0.1, gripper: 0.1 },
		peak_amps: held ? 0.25 : 0.1,
		torque: engaged,
	});
	const carry = () => {
		if (!held) return;
		const t = tip(pose);
		const f = fwd(pose.pan);
		const l = left(pose.pan);
		box = { ...box, x: t.x + held.ahead * f.x - held.right * l.x, y: t.y + held.ahead * f.y - held.right * l.y, z: t.z + held.up };
	};
	const truth = (): FakeTruth => ({ box: { ...box }, held: held !== null, tip: tip(pose) });

	const frame = (cam: Cam): Frame => ({
		cam, t_capture: Date.now() / 1000, age_ms: 5, sha256: FRAME_SHA, width: 320, height: 240, mime: "image/jpeg", data_b64: FRAME.toString("base64"), upright: true,
	});

	const moveTo = (target: Pose, capPan: boolean): MotionResult => {
		const before = state();
		const end = tip(target);
		if (target.reach < 0.05) throw new Refusal(422, "unreachable", `no joints put the grasp point at reach ${target.reach.toFixed(3)} m`);
		if (end.z < TABLE_CLEARANCE_M) throw new Refusal(422, "out_of_envelope", `the grasp point would end ${end.z.toFixed(3)} m above the table, under the ${TABLE_CLEARANCE_M} m floor; nothing moved`);
		if (target.reach > maxReach) throw new Refusal(422, "out_of_envelope", `the grasp point would end at reach ${target.reach.toFixed(3)} m, past max_reach_m ${maxReach}; nothing moved`);
		const turn = target.pan - pose.pan;
		let stopped: string | undefined;
		if (capPan && Math.abs(turn) > MAX_COMMAND_DEG) {
			target = { ...target, pan: pose.pan + Math.sign(turn) * MAX_COMMAND_DEG };
			stopped = `capped at ${MAX_COMMAND_DEG}° per command (${Math.abs(turn).toFixed(0)}° asked)`;
		}
		pose = target;
		carry();
		return { ok: !stopped, ...(stopped ? { stopped } : {}), before, after: state(), ms: 5 };
	};

	const grasp = (to: unknown): MotionResult => {
		const before = state();
		if (to === "open") {
			gripper = Math.max(OPEN_PCT, gripper);
			grip = "open";
			if (held) box = { ...box, z: box.h / 2 };
			held = null;
		} else if (to === "close") {
			const t = tip(pose);
			const inside = Math.abs(t.x - box.x) <= box.w / 2 + JAW_REACH_MARGIN_M && Math.abs(t.y - box.y) <= box.d / 2 + JAW_REACH_MARGIN_M && Math.abs(t.z - box.z) <= box.h / 2;
			if (inside) {
				const f = fwd(pose.pan);
				const l = left(pose.pan);
				const dx = box.x - t.x;
				const dy = box.y - t.y;
				held = { ahead: dx * f.x + dy * f.y, right: -(dx * l.x + dy * l.y), up: box.z - t.z };
				gripper = round((100 * box.d) / JAW_GAP_AT_FULL_OPEN_M);
				grip = "holding";
				poses.last_grasp = { ...pose };
			} else {
				gripper = 0;
				grip = "empty";
			}
		} else throw new Refusal(400, "bad_request", "grip `to` is open or close");
		return { ok: true, before, after: state(), ms: 5 };
	};

	const act = (motion: any): MotionResult => {
		if (!motion || typeof motion !== "object") throw new Refusal(400, "bad_request", "body is {motion: Motion}");
		if (motion.kind === "remember") {
			if (typeof motion.name !== "string" || !motion.name) throw new Refusal(400, "bad_request", "a pose name is a non-empty string");
			poses[motion.name] = { ...pose };
			const s = state();
			return { ok: true, before: s, after: s, ms: 0 };
		}
		if (!(opts.allowMotion ?? true)) throw new Refusal(403, "motion_not_allowed", "allow_motion = false in valet.toml: only remember works");
		if (!["delta", "pose", "grip"].includes(motion.kind)) throw new Refusal(400, "bad_request", `unknown motion kind ${JSON.stringify(motion.kind)}: delta, pose, grip or remember`);
		if (!engaged) {
			const s = state();
			return { ok: false, refused: "the arm is not engaged (torque off, or armd just started): POST /hold first", before: s, after: s, ms: 0 };
		}
		if (motion.kind === "grip") return grasp(motion.to);
		if (motion.kind === "pose") {
			const target = poses[motion.name];
			if (!target) throw new Refusal(404, "not_found", `no remembered pose named ${JSON.stringify(motion.name)}`);
			return moveTo({ ...target }, true);
		}
		const n = (k: string): number => {
			const v = motion[k] ?? 0;
			if (typeof v !== "number" || !Number.isFinite(v)) throw new Refusal(400, "bad_request", `${k} must be a number`);
			return v;
		};
		return moveTo({ pan: pose.pan + n("pan_deg"), reach: pose.reach + n("reach_m"), height: pose.height + n("height_m"), pitch: pose.pitch + n("pitch_deg"), roll: pose.roll + n("roll_deg") }, true);
	};

	const route = async (req: IncomingMessage): Promise<unknown> => {
		const path = new URL(req.url ?? "/", "http://double").pathname;
		calls.push(`${req.method} ${path}`);
		if (failure) throw new Error(failure);
		const host = (req.headers.host ?? "").replace(/:\d+$/, "");
		if (!LOOPBACK.has(host)) throw new Refusal(403, "forbidden_host", `Host ${JSON.stringify(host)} is not loopback`);
		let body: any = {};
		if (req.method === "POST") {
			if ((req.headers["content-type"] ?? "").split(";")[0]!.trim() !== "application/json") throw new Refusal(400, "bad_request", "POST bodies are JSON: send Content-Type: application/json");
			const raw = await readBody(req);
			try {
				body = raw.trim() ? JSON.parse(raw) : {};
			} catch {
				throw new Refusal(400, "bad_request", "body is not JSON");
			}
		}
		const key = `${req.method} ${path}`;
		if (key === "GET /state") return state();
		if (key === "GET /poses") return Object.fromEntries(Object.entries(poses).map(([name, p]) => [name, { joints: joints(p) }]));
		if (key === "POST /frames") {
			if (!Array.isArray(body.cams)) throw new Refusal(400, "bad_request", 'cams is a list of camera names, e.g. ["wrist", "side"]');
			const frames: Partial<Record<Cam, Frame>> = {};
			const missing: Partial<Record<Cam, string>> = {};
			for (const cam of body.cams as Cam[]) {
				if (cams.includes(cam)) frames[cam] = frame(cam);
				else missing[cam] = "no such camera";
			}
			return { frames, missing };
		}
		if (key === "POST /act") return act(body.motion);
		if (key === "POST /locate") return { hit: true, xyz_m: [0.2, 0.05, body.height_m ?? 0], pan_deg: -14.0 + (body.v ?? 0.5) * 2, reach_m: 0.2, height_m: body.height_m ?? 0, ray: { origin_m: [0.15, 0, 0.25], dir: [0.3, 0.1, -0.95] } };
		if (key === "POST /segment") {
			if (!Array.isArray(body.prompts) || body.prompts.length === 0) throw new Refusal(400, "bad_request", "prompts is a list of 1 to 8 short object names");
			const { data_b64: _, ...meta } = frame(body.cam ?? "wrist");
			const seen = () => ({
				found: true, score: 0.9, count: 1, whole: true, touches: [], area: 0.04, bbox_px: [128, 72, 192, 168], box: { left: 0.4, top: 0.3, right: 0.6, bottom: 0.7 },
				centroid: { u: 0.5, v: 0.5 }, top: { u: 0.5, v: 0.3 }, bottom: { u: 0.5, v: 0.7 }, left: { u: 0.4, v: 0.5 }, right: { u: 0.6, v: 0.5 },
			});
			const results = Object.fromEntries((body.prompts as string[]).map((p) => [p, p.includes("unicorn") ? { found: false, score: 0.04, count: 0 } : seen()]));
			return { frame: meta, size: body.size ?? 504, ms: 5, results };
		}
		if (key === "POST /triangulate") return { ok: true, xyz_m: [0.2, 0.05, 0.07], pan_deg: -14, reach_m: 0.2, height_m: 0.07, gap_m: 0.004, angle_deg: 18 };
		if (key === "POST /hold") {
			engaged = true;
			return state();
		}
		if (key === "POST /relax") {
			engaged = false;
			return state();
		}
		if (key === "GET /fake/truth") return truth();
		if (key === "POST /fake/reset") {
			const spec = { ...DEFAULT_BOX, ...(body.box ?? {}) };
			box = { ...spec, z: spec.h / 2 };
			pose = { ...START };
			gripper = OPEN_PCT;
			grip = "unknown";
			held = null;
			engaged = true;
			return truth();
		}
		throw new Refusal(404, "not_found", `no route ${key}`);
	};

	const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
		let status = 200;
		let out: unknown;
		try {
			out = await route(req);
		} catch (e) {
			status = e instanceof Refusal ? e.status : 500;
			out = { error: (e as Error).message, kind: e instanceof Refusal ? e.kind : "internal" };
		}
		res.writeHead(status, { "Content-Type": "application/json" });
		res.end(JSON.stringify(out));
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	return {
		url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
		calls,
		truth,
		fail: (detail) => {
			failure = detail;
		},
		close: () =>
			new Promise<void>((resolve) => {
				server.close(() => resolve());
				server.closeAllConnections();
			}),
	};
}

function readBody(req: IncomingMessage): Promise<string> {
	return new Promise((resolve, reject) => {
		let data = "";
		req.on("data", (c) => (data += c));
		req.on("end", () => resolve(data));
		req.on("error", reject);
	});
}

function round(v: number): number {
	return Math.round(v * 100) / 100;
}
