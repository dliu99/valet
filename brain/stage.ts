import { setTimeout as delay } from "node:timers/promises";
import { ArmError, CAMS, type ArmClient, type ArmState, type Frames, type Motion, type MotionResult } from "./arm.ts";
import type { Answers, DecisionRequest, Exchange } from "./decision.ts";
import type { RecordedModel, Recorder, TickData, TickPlan } from "./record.ts";
import { compile, read, RECENT, SURE, toMotion, UNSURE_HANDBACK, type Reading, type Scene, type Step, type Task } from "./task.ts";

export type Exit = "done" | "stuck" | "unsure" | "budget" | "fault" | "interrupted";

export interface Report {
	exit: Exit;
	reason: string;
	steps: Step[];
	arm: ArmState | null;
	frames: Frames;
}

export interface TaskDeps {
	arm: ArmClient;
	model: RecordedModel;
	recorder: Recorder;
	view: string;
}

export const MAX_FAILED_TICKS = 3;
export const RETRY_AFTER_FAILURE_MS = 400;

export async function runTask(task: Task, deps: TaskDeps, signal: AbortSignal, before: readonly Step[] = []): Promise<Report> {
	const steps: Step[] = [];
	const recent = () => [...before, ...steps].slice(-RECENT);
	const started = Date.now();
	let failures = 0;
	let unsure = 0;
	let tick = 0;
	let arm: ArmState | null = null;
	const end = async (exit: Exit, reason: string): Promise<Report> => {
		if (exit === "interrupted") await deps.arm.hold().catch(() => undefined);
		const last = await finalLook(deps.arm);
		return { exit, reason, steps, arm: last.arm ?? arm, frames: last.frames };
	};

	try {
		arm = await deps.arm.hold();
	} catch (e) {
		return end("fault", `armd would not hold: ${(e as Error).message}`);
	}
	for (;;) {
		if (signal.aborted) return end("interrupted", "stopped from outside");
		if (Date.now() - started >= task.seconds * 1000) return end("budget", `${task.seconds} s spent`);
		if (failures >= MAX_FAILED_TICKS) return end("fault", `${failures} ticks in a row failed`);
		if (failures > 0) await delay(RETRY_AFTER_FAILURE_MS, undefined, { signal }).catch(() => undefined);
		tick++;

		let scene: Scene;
		let framesMs: number;
		try {
			const t0 = performance.now();
			arm = await deps.arm.state();
			const look = await deps.arm.frames(["wrist"], deps.model.caps.maxSide);
			framesMs = Math.round(performance.now() - t0);
			if (!look.frames.wrist) {
				failures++;
				continue;
			}
			scene = { arm, frame: look.frames.wrist, recent: recent(), view: deps.view };
		} catch (e) {
			if (e instanceof ArmError && e.kind !== "network") return end("fault", `armd: ${e.message}`);
			failures++;
			continue;
		}
		let plan: TickPlan | undefined;
		try {
			await deps.model.decide(compile(task, scene), {
				role: "tick",
				task: task.id,
				tick,
				facts: { arm: scene.arm, recent: scene.recent, framesMs },
				plan: (a) => (plan = planTick(a, scene.arm)),
			});
		} catch {
			failures++;
			continue;
		}
		failures = 0;
		const { reading, motion, step } = plan!;
		if (reading.p < SURE) {
			steps.push({ tick, action: reading.action, p: reading.p, step: null, did: "unsure; did nothing" });
			unsure++;
			if (unsure >= UNSURE_HANDBACK) return end("unsure", `Jev was unsure ${unsure} ticks in a row (last: ${reading.action} at p ${reading.p.toFixed(2)})`);
			continue;
		}
		unsure = 0;
		const confirmed = reading.action === "done" && scene.recent.at(-1)?.action === "done" && scene.recent.at(-1)!.p >= SURE;
		let did = reading.action === "done" ? (confirmed ? "done" : "claimed done; looking again") : reading.action;
		if (motion) {
			const carried = await carryOut(deps, task.id, tick, motion);
			if (carried.fault) return end("fault", carried.fault);
			did = carried.did;
		}
		steps.push({ tick, action: reading.action, p: reading.p, step, did });
		if (confirmed) return end("done", "Jev judged the goal done twice in a row");
		if (reading.action === "stuck") return end("stuck", `Jev chose stuck (p ${reading.p.toFixed(2)})`);
	}
}

function planTick(answers: Answers, arm: ArmState): TickPlan {
	const reading = read(answers);
	return { reading, ...toMotion(reading.action, answers, arm) };
}

async function carryOut(deps: TaskDeps, task: string, tick: number, motion: Motion): Promise<{ did: string; fault?: string }> {
	const t0 = performance.now();
	let result: MotionResult | null = null;
	try {
		result = await deps.arm.act(motion);
	} catch (e) {
		const error = e instanceof ArmError ? `${e.kind}: ${e.message}` : (e as Error).message;
		await deps.recorder.motion({ task, tick, motion, result: null, error, ms: Math.round(performance.now() - t0) });
		if (e instanceof ArmError && e.refusal) return { did: `refused: ${error}` };
		return { did: `failed: ${error}`, fault: `armd: ${error}` };
	}
	await deps.recorder.motion({ task, tick, motion, result: result as never, error: null, ms: Math.round(performance.now() - t0) });
	if (result.ok) return { did: "moved" };
	if (result.stopped) return { did: `stopped: ${result.stopped}` };
	return { did: `refused: ${result.refused ?? "no reason given"}` };
}

async function finalLook(arm: ArmClient): Promise<{ arm: ArmState | null; frames: Frames }> {
	try {
		const [state, look] = await Promise.all([arm.state(), arm.frames(CAMS)]);
		return { arm: state, frames: look.frames };
	} catch {
		return { arm: null, frames: {} };
	}
}

export interface Asked {
	request: DecisionRequest;
	answers: Answers;
	reading: Reading;
	motion: Motion | null;
}

export async function askTask(task: Task, deps: TaskDeps): Promise<Asked> {
	const arm = await deps.arm.state();
	const look = await deps.arm.frames(["wrist"], deps.model.caps.maxSide);
	if (!look.frames.wrist) throw new Error(`no wrist frame (${look.missing.wrist ?? "missing"})`);
	return decideOnce(task, { arm, frame: look.frames.wrist, recent: [], view: deps.view }, deps.model, "ask");
}

export async function reask(task: Task, tick: TickData, deps: Omit<TaskDeps, "arm">, onExchange?: (e: Exchange) => void): Promise<Asked> {
	const ref = tick.frames.find((f) => f.cam === "wrist");
	const frame = ref && deps.recorder.loadFrame(ref.sha256, "wrist", ref);
	if (!frame) throw new Error(`tick ${tick.tick} has no wrist frame in ${deps.recorder.framesDir}`);
	const scene: Scene = { arm: tick.arm as unknown as ArmState, frame, recent: (tick.recent ?? []) as unknown as Step[], view: deps.view };
	return decideOnce(task, scene, deps.model, "replay", tick.tick ?? undefined, onExchange);
}

async function decideOnce(task: Task, scene: Scene, model: RecordedModel, role: "ask" | "replay", tick?: number, onExchange?: (e: Exchange) => void): Promise<Asked> {
	const request = compile(task, scene);
	let plan: TickPlan | undefined;
	const answers = await model.decide(request, {
		role,
		task: task.id,
		tick,
		exchange: onExchange,
		facts: { arm: scene.arm, recent: scene.recent },
		plan: (a) => (plan = planTick(a, scene.arm)),
	});
	return { request, answers, reading: plan!.reading, motion: plan!.motion };
}
