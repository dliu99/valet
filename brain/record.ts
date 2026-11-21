import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import type { Models } from "@earendil-works/pi-ai/models";
import { createModels } from "@earendil-works/pi-ai/models";
import { createRegistry, defineDoc, defineEntry, Harness, type ConversationId, type EntryRecord, type Registry } from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";
import type { ArmState, Frame, Motion } from "./arm.ts";
import type { Answers, DecideCtx, DecisionModel, DecisionRequest, Exchange, ImageName, Json, JsonObject } from "./decision.ts";
import type { Reading, Task } from "./task.ts";

export const ctx = BACKGROUND_CONTEXT;

export interface FrameRef {
	[key: string]: Json;
	cam: ImageName;
	sha256: string;
	width: number;
	height: number;
}

export interface TickData {
	[key: string]: Json;
	task: string | null;
	tick: number | null;
	role: DecideCtx["role"];
	decider: string;
	t: number;
	arm: Json;
	recent: Json;
	state: Json;
	questions: Json;
	frames: FrameRef[];
	wire: { request: Json; response: Json };
	answers: Json;
	reading: Json;
	motion: Json;
	step: string | null;
	ms: { frames: number | null; model: number };
	usage: Json;
	error: string | null;
}

export interface MotionData {
	[key: string]: Json;
	task: string;
	tick: number;
	motion: Json;
	result: Json;
	error: string | null;
	ms: number;
}

/** One step toward a goal (brain/director.ts): the skill Jev chose, or done, stuck or unsure, and what came of it. */
export interface StepData {
	[key: string]: Json;
	task: string;
	step: number;
	goal: string;
	what: string;
	skill: string;
	amount: number | null;
	unit: string | null;
	/** Jev's probability for the choice; null when a person ran the skill directly. */
	p: number | null;
	ok: boolean;
	why: string;
	result: Json;
	log: string[];
	held: Json;
	ms: number;
}

export interface LlmRequestData {
	[key: string]: Json;
	model: string;
	api: string;
	body: Json;
	t: number;
}

export const TickEntry = defineEntry<TickData>("valet.tick");
export const MotionEntry = defineEntry<MotionData>("valet.motion");
export const StepEntry = defineEntry<StepData>("valet.step");
export const LlmRequestEntry = defineEntry<LlmRequestData>("valet.llm_request");

export type RunKind = "run" | "do" | "skill" | "ask" | "replay";
export type RunStatus = "running" | "done" | "stopped" | "failed";

export interface RunState {
	[key: string]: Json;
	kind: RunKind;
	goal: string;
	startedAt: number;
	endedAt: number | null;
	status: RunStatus;
	summary: string | null;
	planner: string;
	decider: string;
	tasks: Json[];
	programs: Json[];
	/** Claude's verdicts at the checkpoints of a `do` run. */
	checkpoints: Json[];
}

export const RunDoc = defineDoc<RunState>({
	kind: "valet.run",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ kind: "run", goal: "", startedAt: 0, endedAt: null, status: "running", summary: null, planner: "", decider: "", tasks: [], programs: [], checkpoints: [] }),
	checkpointWhen: () => true,
});

export interface RunIndexState {
	[key: string]: Json;
	runs: { conversationId: number; kind: RunKind; goal: string; startedAt: number }[];
}

export const RunIndex = defineDoc<RunIndexState>({
	kind: "valet.runs",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ runs: [] }),
	checkpointWhen: () => true,
});

export interface Store {
	harness: Harness;
	runsDir: string;
	framesDir: string;
	readOnly: boolean;
	close(): Promise<void>;
}

export async function openStore(runsDir: string, opts: { readOnly?: boolean; models?: Models; registry?: Registry } = {}): Promise<Store> {
	const framesDir = join(runsDir, "frames");
	mkdirSync(framesDir, { recursive: true });
	const storage = await openNodeSqliteStorage(join(runsDir, "valet.sqlite"));
	const harness = await Harness.open(
		storage,
		{ models: opts.models ?? createModels(), registry: opts.registry ?? createRegistry(), readOnly: opts.readOnly },
		ctx,
	);
	return { harness, runsDir, framesDir, readOnly: Boolean(opts.readOnly), close: () => harness.close(ctx) };
}

export async function startRun(store: Store, init: Pick<RunState, "kind" | "goal" | "planner" | "decider">): Promise<ConversationId> {
	const conv = await store.harness.createConversation({ ownership: { kind: "ownerless" } }, ctx);
	const startedAt = Date.now();
	await conv.commit(async (tx) => {
		Object.assign(await tx.doc(RunDoc, conv.id), { ...init, startedAt, status: "running" });
	}, ctx);
	const root = await store.harness.root(ctx);
	await root.commit(async (tx) => {
		(await tx.doc(RunIndex, root.id)).runs.push({ conversationId: conv.id, kind: init.kind, goal: init.goal, startedAt });
	}, ctx);
	return conv.id;
}

export async function updateRun(store: Store, id: ConversationId, change: (run: RunState) => void): Promise<void> {
	const conv = await store.harness.conversation(id, ctx);
	if (!conv) throw new Error(`no run ${id}`);
	await conv.commit(async (tx) => change((await tx.doc(RunDoc, id)) as RunState), ctx);
}

export async function addTask(store: Store, id: ConversationId, task: Task): Promise<void> {
	await updateRun(store, id, (run) => {
		run.tasks.push(structuredClone(task) as unknown as Json);
	});
}

export async function addProgram(store: Store, id: ConversationId, program: JsonObject): Promise<void> {
	await updateRun(store, id, (run) => {
		(run.programs ??= []).push(structuredClone(program));
	});
}

export async function runEntries(store: Store, id: ConversationId): Promise<EntryRecord[]> {
	const conv = await store.harness.conversation(id, ctx);
	if (!conv) throw new Error(`no run ${id}`);
	const entries: EntryRecord[] = [];
	let cursor;
	do {
		const page = await conv.entries({ order: "ascending" }, 500, cursor, ctx);
		entries.push(...page.items);
		cursor = page.next;
	} while (cursor);
	return entries;
}

export async function recordedTicks(store: Store, id: ConversationId): Promise<TickData[]> {
	return (await runEntries(store, id)).filter((e) => e.kind === TickEntry.kind && (e.data as TickData).role === "tick").map((e) => e.data as TickData);
}

export interface Sink {
	append(kind: string, data: JsonObject): Promise<void>;
}

export function conversationSink(store: Store, id: ConversationId): Sink {
	return {
		async append(kind, data) {
			const conv = await store.harness.conversation(id, ctx);
			if (!conv) throw new Error(`no conversation ${id}`);
			await conv.commit(async (tx) => {
				await tx.appendEntry(id, { kind, data });
			}, ctx);
		},
	};
}

export function fileSink(path: string): Sink {
	mkdirSync(dirname(path), { recursive: true });
	return {
		async append(kind, data) {
			appendFileSync(path, `${JSON.stringify({ kind, data })}\n`);
		},
	};
}

/** What the caller made of the answers, kept with the decision. */
export interface DecisionPlan {
	reading: { action: string; p: number };
	motion: Motion | null;
	step: string | null;
}

export interface TickPlan extends DecisionPlan {
	reading: Reading;
}

export interface TickFacts {
	arm: ArmState;
	recent: readonly object[];
	framesMs?: number;
}

export interface RecordCtx {
	plan?: (answers: Answers) => DecisionPlan;
	facts?: TickFacts;
}

export class Recorder {
	readonly framesDir: string;
	readonly sink: Sink;

	constructor(framesDir: string, sink: Sink) {
		this.framesDir = framesDir;
		this.sink = sink;
		mkdirSync(framesDir, { recursive: true });
	}

	saveBytes(bytes: Buffer): string {
		const sha = createHash("sha256").update(bytes).digest("hex");
		const file = join(this.framesDir, `${sha}.jpg`);
		if (!existsSync(file)) writeFileSync(file, bytes);
		return sha;
	}

	saveFrame(frame: Frame): FrameRef {
		return { cam: frame.cam, sha256: this.saveBytes(Buffer.from(frame.data_b64, "base64")), width: frame.width, height: frame.height };
	}

	loadFrame(sha: string, cam: Frame["cam"], size: { width: number; height: number }): Frame | undefined {
		const file = join(this.framesDir, `${sha}.jpg`);
		if (!existsSync(file)) return undefined;
		const bytes = readFileSync(file);
		return { cam, t_capture: 0, age_ms: 0, sha256: sha, ...size, mime: "image/jpeg", data_b64: bytes.toString("base64"), upright: true };
	}

	redact(value: unknown): Json {
		if (typeof value === "string") {
			const image = imageBytes(value);
			return image ? this.saveBytes(image) : value;
		}
		if (Array.isArray(value)) return value.map((v) => this.redact(v));
		if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, this.redact(v)]));
		return (value ?? null) as Json;
	}

	tick(data: TickData): Promise<void> {
		return this.sink.append(TickEntry.kind, data);
	}

	motion(data: MotionData): Promise<void> {
		return this.sink.append(MotionEntry.kind, data);
	}

	step(data: StepData): Promise<void> {
		return this.sink.append(StepEntry.kind, data);
	}

	llmRequest(model: { provider: string; id: string; api: string }, body: unknown): Promise<void> {
		return this.sink.append(LlmRequestEntry.kind, { model: `${model.provider}/${model.id}`, api: model.api, body: this.redact(body), t: Date.now() });
	}
}

export interface RecordedModel extends DecisionModel {
	decide(req: DecisionRequest, c: DecideCtx & RecordCtx): Promise<Answers>;
}

export function recorded(model: DecisionModel, recorder: Recorder): RecordedModel {
	return {
		id: model.id,
		caps: model.caps,
		async decide(req, c) {
			const frames = Object.entries(req.images).flatMap(([name, f]) => (f ? [{ ...recorder.saveFrame(f), cam: name as ImageName }] : []));
			let exchange: Exchange | null = null;
			const t = Date.now();
			const t0 = performance.now();
			const entry = (outcome: Pick<TickData, "answers" | "reading" | "motion" | "step" | "error">): TickData => ({
				task: c.task ?? null,
				tick: c.tick ?? null,
				role: c.role,
				decider: model.id,
				t,
				arm: (c.facts?.arm ?? null) as Json,
				recent: (c.facts?.recent ?? []) as unknown as Json,
				state: req.state,
				questions: req.questions as unknown as Json,
				frames,
				wire: { request: recorder.redact(exchange?.request ?? null), response: recorder.redact(exchange?.response ?? null) },
				ms: { frames: c.facts?.framesMs ?? null, model: Math.round(performance.now() - t0) },
				usage: (exchange?.usage ?? null) as Json,
				...outcome,
			});
			let answers: Answers;
			try {
				answers = await model.decide(req, {
					role: c.role,
					task: c.task,
					tick: c.tick,
					exchange: (e) => {
						exchange = e;
						c.exchange?.(e);
					},
				});
			} catch (e) {
				await recorder.tick(entry({ answers: null, reading: null, motion: null, step: null, error: (e as Error).message }));
				throw e;
			}
			const plan = c.plan?.(answers);
			await recorder.tick(
				entry({
					answers: answers as unknown as Json,
					reading: (plan?.reading ?? null) as unknown as Json,
					motion: (plan?.motion ?? null) as Json,
					step: plan?.step ?? null,
					error: null,
				}),
			);
			return answers;
		},
	};
}

function imageBytes(s: string): Buffer | undefined {
	if (s.length < 256) return undefined;
	const data = s.startsWith("data:image/") ? s.slice(s.indexOf(",") + 1) : s;
	if (!/^[A-Za-z0-9+/]+={0,2}$/.test(data.slice(0, 4096))) return undefined;
	const bytes = Buffer.from(data, "base64");
	const jpeg = bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
	const png = bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47;
	return jpeg || png ? bytes : undefined;
}

