import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { defineExtension, defineTool, GenerationTask, hook, LiveDoc, section, type ConversationId, type Extension } from "@earendil-works/pi-durable";
import { CAMS, type ArmClient, type Frames } from "./arm.ts";
import type { Config } from "./config.ts";
import type { DecisionModel } from "./decision.ts";
import { armSummary, runProgram } from "./program.ts";
import { addProgram, conversationSink, ctx, recorded, Recorder, RunDoc, startRun, updateRun, type RunState, type Store } from "./record.ts";
import { WRIST_CONVENTION } from "./task.ts";

export interface PlannerServices {
	config: Config;
	arm: ArmClient;
	decider: DecisionModel;
	store: Store;
	person(question: string): Promise<string>;
}

const DEFAULT_SECONDS = 90;
const LESSONS_SHOWN = 6;
const SKILL_CHARS = 3000;

export interface Memory {
	lessons: string[];
	skill: { goal: string; programs: string[] } | null;
}

export function readMemory(runsDir: string): Memory {
	const lines = (file: string) => (existsSync(file) ? readFileSync(file, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : []);
	const lessons = lines(join(runsDir, "lessons.jsonl")).slice(-LESSONS_SHOWN).map((l: { lesson: string; met: boolean }) => `${l.met ? "(met)" : "(not met)"} ${l.lesson}`);
	const last = lines(join(runsDir, "skills.jsonl")).at(-1) as { goal: string; programs: string[] } | undefined;
	return { lessons, skill: last ? { goal: last.goal, programs: last.programs } : null };
}

function memorySection(m: Memory): string {
	const parts: string[] = [];
	if (m.lessons.length) parts.push(`Lessons from earlier runs on this arm, newest last (they may be about other objects or places; measure this run's):\n${m.lessons.map((l) => `- ${l}`).join("\n")}`);
	if (m.skill) {
		const code = m.skill.programs.join("\n// --- next program\n").slice(0, SKILL_CHARS);
		parts.push(`Programs from the last run that met its goal ("${m.skill.goal}"); adapt, don't copy blindly:\n\`\`\`js\n${code}\n\`\`\``);
	}
	return parts.join("\n\n");
}

export function plannerExtension(svc: PlannerServices): Extension {
	const recorderFor = (id: ConversationId) => new Recorder(svc.store.framesDir, conversationSink(svc.store, id));
	const text = (value: unknown): TextContent => ({ type: "text", text: typeof value === "string" ? value : JSON.stringify(value, null, 1) });

	return defineExtension({
		name: "valet-planner",
		sections: [
			section("preamble", () => PREAMBLE, { tag: false }),
			section("api", () => apiSection(svc.config, svc.decider)),
			section("memory", () => memorySection(readMemory(svc.store.runsDir))),
		],
		hooks: [
			hook(GenerationTask, {
				beforeProviderRequest: async ({ model, payload }, api) => {
					await recorderFor(api.conversationId).llmRequest(model, payload);
					return undefined;
				},
			}),
		],
		tools: [
			defineTool({
				name: "look",
				description: "The cameras at full resolution and the arm's state.",
				parameters: { type: "object", properties: {}, additionalProperties: false },
				execute: async (_args, api) => {
					const [arm, look] = await Promise.all([svc.arm.state(), svc.arm.frames(CAMS)]);
					return { output: [text({ arm: armSummary(arm), frames: savedFrames(recorderFor(api.conversationId), look.frames), missing: look.missing }), ...images(look.frames)] };
				},
			}),
			defineTool({
				name: "program",
				description: `Run a short async JavaScript program against the arm API in the system prompt and get back its log, any error, the arm's state and the final pictures. It stops after \`seconds\` (default ${DEFAULT_SECONDS}).`,
				parameters: {
					type: "object",
					properties: { code: { type: "string", description: "the body of an async function; top-level await works" }, seconds: { type: "number" } },
					required: ["code"],
					additionalProperties: false,
				},
				execute: async (args, api, context) => {
					const { code, seconds = DEFAULT_SECONDS } = args as { code: string; seconds?: number };
					const id = api.conversationId;
					const n = (await svc.store.harness.snapshot(RunDoc, id, ctx))?.programs?.length ?? 0;
					const recorder = recorderFor(id);
					const result = await runProgram(`p${n}`, code, { arm: svc.arm, model: recorded(svc.decider, recorder), recorder, view: svc.config.viewpoints.wrist }, context.abortSignal ?? new AbortController().signal, seconds);
					const frames = savedFrames(recorder, result.frames);
					await addProgram(svc.store, id, { id: `p${n}`, code, ok: result.ok, error: result.error, log: result.log, ms: result.ms, frames });
					const summary = { ok: result.ok, error: result.error, log: result.log, arm: result.arm ? armSummary(result.arm) : null, seconds: Math.round(result.ms / 100) / 10 };
					return { output: [text(summary), ...images(result.frames)], ...(result.ok ? {} : { isError: true }) };
				},
			}),
			defineTool({
				name: "finish",
				description: "End the run with a short summary of what was done and whether the goal was met.",
				parameters: {
					type: "object",
					properties: {
						summary: { type: "string" },
						met: { type: "boolean", description: "whether the goal was met (default true)" },
						lesson: { type: "string", description: "one or two sentences for the next run on this arm: what worked or what to avoid" },
					},
					required: ["summary"],
					additionalProperties: false,
				},
				execute: async (args, api) => {
					const { summary, met = true, lesson } = args as { summary: string; met?: boolean; lesson?: string };
					let goal = "";
					let programs: string[] = [];
					await updateRun(svc.store, api.conversationId, (run) => {
						run.status = met ? "done" : "stopped";
						run.summary = summary;
						run.endedAt = Date.now();
						goal = run.goal;
						programs = (run.programs ?? []).flatMap((p) => ((p as { ok?: boolean }).ok ? [String((p as { code?: string }).code)] : [])).slice(-4);
					});
					if (lesson?.trim()) appendFileSync(join(svc.store.runsDir, "lessons.jsonl"), `${JSON.stringify({ t: Date.now(), goal, met, lesson: lesson.trim() })}\n`);
					if (met && programs.length) appendFileSync(join(svc.store.runsDir, "skills.jsonl"), `${JSON.stringify({ t: Date.now(), goal, programs })}\n`);
					return { output: [text("Recorded.")], control: { terminate: true } };
				},
			}),
			defineTool({
				name: "ask_person",
				description: "Ask the person running the robot something only they can do or know; returns their answer.",
				parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"], additionalProperties: false },
				execute: async (args) => ({ output: [text(await svc.person((args as { text: string }).text))] }),
			}),
		],
	});
}

export async function planRun(svc: PlannerServices, goal: string, planner: { provider: string; modelId: string }, onStart?: (id: ConversationId) => void): Promise<{ id: ConversationId; run: RunState }> {
	const id = await startRun(svc.store, { kind: "run", goal, planner: `${planner.provider}/${planner.modelId}`, decider: svc.decider.id });
	onStart?.(id);
	const conv = (await svc.store.harness.conversation(id, ctx))!;
	await conv.configure({ model: planner, maxTurns: 40 }, ctx);
	const submission = await conv.submit({ type: "input", content: `Goal: ${goal}`, requestId: "goal" }, ctx);
	const settled = await submission.wait(ctx);
	const live = await svc.store.harness.snapshot(LiveDoc, id, ctx);
	let run = (await svc.store.harness.snapshot(RunDoc, id, ctx)) as RunState;
	if (run.status === "running") {
		const why = settled.status === "done" ? "the planner answered without calling finish" : `the planner's run ended: ${live?.runFailure?.reason ?? settled.status}`;
		await updateRun(svc.store, id, (r) => {
			r.status = settled.status === "done" ? "stopped" : "failed";
			r.summary = why;
			r.endedAt = Date.now();
		});
		run = (await svc.store.harness.snapshot(RunDoc, id, ctx)) as RunState;
	}
	return { id, run };
}

function images(frames: Frames): ImageContent[] {
	return CAMS.flatMap((cam) => (frames[cam] ? [{ type: "image" as const, data: frames[cam]!.data_b64, mimeType: frames[cam]!.mime }] : []));
}

function savedFrames(recorder: Recorder, frames: Frames): Record<string, string> {
	return Object.fromEntries(Object.entries(frames).flatMap(([cam, f]) => (f ? [[cam, recorder.saveFrame(f).sha256]] : [])));
}

const PREAMBLE = `You control a small robot arm, an SO-101 with a two-jaw gripper and a camera on the wrist, by writing short programs. \
look() to see, then program() a whole attempt at once: find the object, move above it, descend, grip, lift, verify, with checks \
in the code. Each program runs straight through on the arm; you see its log and the final pictures and write the next one. \
Prefer one program per attempt over many small ones. finish() when the goal is met or cannot be; ask_person() only for what \
needs a person.`;

function apiSection(config: Config, decider: DecisionModel): string {
	return `Inside program(), these async functions are in scope (distances in cm, angles in degrees):
- arm() -> {pan_deg, reach_cm, height_cm, pitch_deg, roll_deg, gripper_pct, grip}. reach is the horizontal distance from the base \
axis to the grasp point between the jaw tips; height is the jaw tips' height above the table; pan positive turns toward the arm's \
right; pitch 0 is the gripper horizontal, 90 pointing straight down. grip: open, holding (closed part way on something), empty \
(closed on nothing), jammed, dropped (it was holding, then closed further: the object slipped out), unknown.
- moveTo({pan_deg?, reach_cm?, height_cm?, pitch_deg?, roll_deg?}) puts the grasp point there; fields left out keep their value. \
move({...}) is the same relative to now. pose(name) goes to a remembered pose (overview). All return {result, arm}; result is \
moved or "stopped: ..." (blocked on the way, or ended short of the target). A move the arm cannot reach throws "refused: ...", \
which ends the program unless you catch it; the message says which pitches reach that point.
- grip("open" | "close", pct?) -> {result, grip, arm}. Closing stops when the jaws meet resistance. Opening goes to pct (default 60, \
up to 100); 1-3 cm behind the tips the jaws are about 6.5 cm apart at 45, 8.5 at 60 and 10.5 at 80 (only 3-5 cm behind the tips at 80). The grasp point that pan/reach/height \
describe is the tip of the fixed jaw's inner face; the moving jaw closes toward it from the arm's right, so a gripped object ends up \
centred half its width to the right of the grasp point, and about as far behind the tips as you pushed past it.
- jawsAt({pan_deg, reach_cm, height_cm}, {pitch_deg, width_cm?, depth_cm?}) -> the moveTo pose that puts a point in the middle of \
the open jaws, depth_cm (default 2) behind the tips, approaching at pitch_deg; gap_cm is the opening (default 7.5, for 60). Not a move.
- locate(what, {height_cm?}) asks ${decider.id}, a fast vision model, for a box around \`what\` in the wrist image and returns \
{found, confidence, box, centre, foot}: centre is the box centre placed on a horizontal plane at height_cm (default 0, the table), \
foot is the box's bottom edge on the table. Each is {pan_deg, reach_cm, height_cm} or null. Use the height of the part you aimed at.
- triangulate(a, b) takes two locate() results of the same thing seen from two poses (move at least 5 cm or 10° between them) and \
returns its position and height {ok, pan_deg, reach_cm, height_cm, gap_cm, angle_deg}: no height guess needed. gap_cm is how far \
the two sight lines miss each other; a large gap means the two boxes were not the same spot.
- check(question) -> the probability (0..1) that a yes/no question about the wrist image is yes, from ${decider.id}.
- jev(goal, {notes?, seconds?}) lets ${decider.id} steer in small closed-loop steps from the camera until it judges the goal done, \
gives up or runs out of time; returns {exit, reason, steps}.
- log(...), sleep(ms).
The wrist camera: ${config.viewpoints.wrist}. ${WRIST_CONVENTION} It sees nothing below or behind the jaws; raise the arm and \
tilt down to see more of the table. Positions from locate come from one camera and are rough; check before closing and after lifting.`;
}
