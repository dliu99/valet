export type Cam = "wrist" | "side";
export const CAMS: readonly Cam[] = ["wrist", "side"];

export type Grip = "open" | "holding" | "jammed" | "empty" | "unknown";

export interface ArmState {
	t: number;
	kind: "so101" | "fake" | "sim";
	allow_motion: boolean;
	joints: { shoulder_pan: number; shoulder_lift: number; elbow_flex: number; wrist_flex: number; wrist_roll: number; gripper: number };
	tip: { x: number; y: number; z: number };
	pan_deg: number;
	reach_m: number;
	height_m: number;
	pitch_deg: number;
	roll_deg: number;
	gripper_pct: number;
	grip: Grip;
	amps: Record<string, number>;
	peak_amps: number;
	torque: boolean;
}

export interface Frame {
	cam: Cam;
	t_capture: number;
	age_ms: number;
	sha256: string;
	width: number;
	height: number;
	mime: "image/jpeg";
	data_b64: string;
	upright: boolean;
}

export type Motion =
	| { kind: "delta"; pan_deg?: number; reach_m?: number; height_m?: number; pitch_deg?: number; roll_deg?: number; speed_dps?: number }
	| { kind: "to"; pan_deg?: number; reach_m?: number; height_m?: number; pitch_deg?: number; roll_deg?: number; speed_dps?: number }
	| { kind: "pose"; name: string }
	| { kind: "grip"; to: "open" | "close"; pct?: number }
	| { kind: "remember"; name: string };

export interface MotionResult {
	ok: boolean;
	refused?: string;
	stopped?: string;
	before: ArmState;
	after: ArmState;
	ms: number;
}

export type Frames = Partial<Record<Cam, Frame>>;
export type Poses = Record<string, { joints: Partial<ArmState["joints"]> }>;

export class ArmError extends Error {
	readonly kind: string;
	readonly status: number;
	constructor(message: string, kind: string, status: number) {
		super(message);
		this.kind = kind;
		this.status = status;
	}
	get refusal(): boolean {
		return this.status === 422;
	}
}

export interface Ray {
	origin_m: number[];
	dir: number[];
}
export type Located = ({ hit: true; xyz_m: number[]; pan_deg: number; reach_m: number; height_m: number } | { hit: false; why: string }) & { ray: Ray };
/** A point of the upright frame: u across, v down, each 0..1. */
export interface UV {
	u: number;
	v: number;
}
export type Side = "left" | "right" | "top" | "bottom";
/** SAM 3's best mask for one prompt. `whole` is false when the mask reaches the frame's edge (`touches` says which). */
export type Segment =
	| { found: false; score: number; count: 0 }
	| {
			found: true;
			score: number;
			count: number;
			whole: boolean;
			touches: Side[];
			area: number;
			bbox_px: [number, number, number, number];
			box: { left: number; top: number; right: number; bottom: number };
			centroid: UV;
			top: UV;
			bottom: UV;
			left: UV;
			right: UV;
			mask_png_b64?: string;
	  };
export interface Segmentation {
	frame: Omit<Frame, "data_b64"> & { data_b64?: string };
	size: number;
	ms: number;
	load_ms?: number;
	results: Record<string, Segment>;
}
export interface SegmentOptions {
	cam?: Cam;
	size?: 336 | 504 | 672 | 1008;
	maxSide?: number;
	mask?: boolean;
	frame?: boolean;
}
export const SEGMENT_TIMEOUT_MS = 60_000;

export type Triangulated = { ok: true; xyz_m: number[]; pan_deg: number; reach_m: number; height_m: number; gap_m: number; angle_deg: number } | { ok: false; why: string; angle_deg: number };

export interface ArmClient {
	readonly url: string;
	state(): Promise<ArmState>;
	frames(cams: readonly Cam[], maxSide?: number): Promise<{ frames: Frames; missing: Partial<Record<Cam, string>> }>;
	act(motion: Motion): Promise<MotionResult>;
	hold(): Promise<ArmState>;
	poses(): Promise<Poses>;
	locate(u: number, v: number, aspect: number, heightM: number): Promise<Located>;
	triangulate(a: Ray, b: Ray): Promise<Triangulated>;
	segment(prompts: string[], opts?: SegmentOptions): Promise<Segmentation>;
}

export function armClient(url: string, timeoutMs = 15_000): ArmClient {
	const call = async <T>(method: "GET" | "POST", path: string, body?: unknown, ms = timeoutMs): Promise<T> => {
		let res: Response;
		try {
			res = await fetch(url + path, {
				method,
				headers: { "Content-Type": "application/json" },
				body: method === "POST" ? JSON.stringify(body ?? {}) : undefined,
				signal: AbortSignal.timeout(ms),
			});
		} catch (e) {
			throw new ArmError(`armd at ${url} unreachable: ${(e as Error).message}`, "network", 0);
		}
		const text = await res.text();
		let data: unknown;
		try {
			data = JSON.parse(text);
		} catch {
			throw new ArmError(`armd ${method} ${path}: HTTP ${res.status}, not JSON: ${text.slice(0, 200)}`, "bad_response", res.status);
		}
		if (!res.ok) {
			const err = data as { error?: string; kind?: string };
			throw new ArmError(err.error ?? `HTTP ${res.status}`, err.kind ?? "error", res.status);
		}
		return data as T;
	};
	return {
		url,
		state: () => call("GET", "/state"),
		frames: (cams, maxSide) => call("POST", "/frames", maxSide ? { cams, max_side: maxSide } : { cams }),
		act: (motion) => call("POST", "/act", { motion }),
		hold: () => call("POST", "/hold"),
		poses: () => call("GET", "/poses"),
		locate: (u, v, aspect, heightM) => call("POST", "/locate", { u, v, aspect, height_m: heightM }),
		triangulate: (a, b) => call("POST", "/triangulate", { a, b }),
		segment: (prompts, opts = {}) => {
			const { maxSide, ...rest } = opts;
			return call("POST", "/segment", { prompts, ...rest, ...(maxSide ? { max_side: maxSide } : {}) }, Math.max(timeoutMs, SEGMENT_TIMEOUT_MS));
		},
	};
}
