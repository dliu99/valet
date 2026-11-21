import { existsSync, readFileSync, statSync } from "node:fs";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { join } from "node:path";
import type { MutableModels } from "@earendil-works/pi-ai/models";
import type { ConversationId } from "@earendil-works/pi-durable";
import { armClient } from "./arm.ts";
import { DECIDER_KINDS, type Config, type DeciderKind } from "./config.ts";
import { makeDecider, type DecisionModel, type Exchange } from "./decision.ts";
import { ctx, fileSink, recorded, Recorder, RunDoc, RunIndex, runEntries, TickEntry, type Store, type TickData } from "./record.ts";
import { SKILL_MARGIN, SKILL_SURE } from "./director.ts";
import { CLAIMS, ON_DISAGREE } from "./overseer.ts";
import { LOCATE_MIN_CONFIDENCE } from "./program.ts";
import { CHOICE_TARGET, DEFAULT_CHOICE, readPrompts } from "./prompts.ts";
import type { Skill } from "./skills.ts";
import { reask } from "./stage.ts";
import { validate, type Task } from "./task.ts";

const PAGE = new URL("../ui/index.html", import.meta.url);
const SKILLS_FILE = new URL("./skills.ts", import.meta.url);
/** Page loads within this long of the last live frame share it, so several open tabs ask armd for one frame. */
const LIVE_FRAME_REUSE_MS = 250;
const LIVE_MAX_SIDE = 960;

export interface UIOptions {
	store: Store;
	config: Config;
	models: MutableModels;
	port: number;
}

export async function serveUI(opts: UIOptions): Promise<Server> {
	const { store, config } = opts;
	const frames = new Recorder(store.framesDir, fileSink(join(store.runsDir, "reask.jsonl")));
	const live = liveArm(config.armdUrl);
	const library = skillLibrary();
	const deciders = new Map<DeciderKind, Promise<DecisionModel>>();
	const decider = (kind: DeciderKind) => {
		if (!deciders.has(kind)) deciders.set(kind, makeDecider(kind, { config, models: opts.models }));
		return deciders.get(kind)!;
	};

	const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);
	const server = createServer(async (req, res) => {
		const send = (status: number, body: unknown, type = "application/json", cache = "no-store") => {
			res.writeHead(status, { "Content-Type": type, "Cache-Control": cache });
			res.end(type === "application/json" ? JSON.stringify(body) : (body as Buffer));
		};
		try {
			const host = (req.headers.host ?? "").replace(/:\d+$/, "");
			if (!LOOPBACK_HOSTS.has(host)) return send(403, { error: `Host ${JSON.stringify(host)} is not loopback` });
			if (req.method === "POST" && !(req.headers["content-type"] ?? "").startsWith("application/json")) return send(415, { error: "POST bodies are application/json" });
			const url = new URL(req.url ?? "/", "http://localhost");
			const parts = url.pathname.split("/").filter(Boolean);
			if (req.method === "GET" && url.pathname === "/") return send(200, readFileSync(PAGE), "text/html; charset=utf-8");
			if (req.method === "GET" && parts[0] === "frames" && /^[0-9a-f]{64}\.jpg$/.test(parts[1] ?? "")) {
				const file = join(store.framesDir, parts[1]!);
				return existsSync(file) ? send(200, readFileSync(file), "image/jpeg", "max-age=31536000, immutable") : send(404, { error: "no such frame" });
			}
			if (req.method === "GET" && url.pathname === "/api/config") return send(200, { deciders: DECIDER_KINDS, decider: config.decider.kind, armd: config.armdUrl, sure: SKILL_SURE, margin: SKILL_MARGIN, locateMin: LOCATE_MIN_CONFIDENCE, claims: CLAIMS, onDisagree: ON_DISAGREE });
			if (req.method === "GET" && url.pathname === "/api/skills") return send(200, { ...(await library()), prompts: promptsNow(store.runsDir), defaults: { [CHOICE_TARGET]: DEFAULT_CHOICE } });
			if (req.method === "GET" && url.pathname === "/api/live") {
				const state = await live.state().catch((e: Error) => e);
				return send(200, state instanceof Error ? { armd: live.url, error: state.message } : { armd: live.url, state });
			}
			if (req.method === "GET" && url.pathname === "/api/live/wrist.jpg") {
				const frame = await live.wrist().catch((e: Error) => e);
				return frame instanceof Error ? send(502, { error: frame.message }) : send(200, frame, "image/jpeg");
			}
			if (req.method === "GET" && url.pathname === "/api/runs") return send(200, await listRuns(store));
			if (req.method === "GET" && parts[0] === "api" && parts[1] === "runs" && parts[2]) return send(200, await runDetail(store, frames, Number(parts[2]) as ConversationId));
			if (req.method === "POST" && url.pathname === "/api/reask") {
				const body = JSON.parse(await readBody(req)) as { run: number; entry: string; task: Task; decider: DeciderKind };
				if (!DECIDER_KINDS.includes(body.decider)) return send(400, { error: `decider: one of ${DECIDER_KINDS.join(", ")}` });
				const model = await decider(body.decider);
				const errors = validate(body.task);
				if (errors.length) return send(400, { error: `invalid task: ${errors.join("; ")}` });
				const tick = await findTick(store, body.run as ConversationId, body.entry);
				if (!tick) return send(404, { error: "no such tick" });
				let exchange: Exchange | undefined;
				const result = await reask(body.task, tick, { model: recorded(model, frames), recorder: frames, view: config.viewpoints.wrist }, (e) => (exchange = e));
				return send(200, {
					decider: model.id,
					answers: result.answers,
					reading: result.reading,
					motion: result.motion,
					questions: result.request.questions,
					wire: { request: frames.redact(exchange?.request), response: frames.redact(exchange?.response) },
				});
			}
			send(404, { error: "not found" });
		} catch (e) {
			send(500, { error: (e as Error).message });
		}
	});
	return new Promise((resolve) => server.listen(opts.port, "127.0.0.1", () => resolve(server)));
}

/** armd as the page sees it: the arm's state and the newest wrist frame, read only (never /act, /hold or /relax). */
function liveArm(url: string) {
	const arm = armClient(url, 2500);
	let last: { at: number; bytes: Promise<Buffer> } | undefined;
	return {
		url,
		state: () => arm.state(),
		wrist(): Promise<Buffer> {
			if (!last || Date.now() - last.at > LIVE_FRAME_REUSE_MS) {
				const bytes = arm.frames(["wrist"], LIVE_MAX_SIDE).then((look) => {
					if (!look.frames.wrist) throw new Error(`no wrist frame (${look.missing.wrist ?? "missing"})`);
					return Buffer.from(look.frames.wrist.data_b64, "base64");
				});
				const mine = { at: Date.now(), bytes };
				last = mine;
				bytes.catch(() => {
					if (last === mine) last = undefined;
				});
			}
			return last.bytes;
		},
	};
}

interface LibrarySkill {
	name: string;
	says: string;
	amount: { says: string; unit: string; levels: number[] } | null;
	/** The body of the skill's `ready`, in terms of the facts; null when it is always offered. */
	ready: string | null;
}

/** The skill library as brain/skills.ts has it now, re-read when the file changes, so the page shows cards being edited. */
function skillLibrary() {
	let loaded: { mtime: number; skills: LibrarySkill[]; error: string | null } | undefined;
	return async () => {
		const mtime = statSync(SKILLS_FILE).mtimeMs;
		if (loaded?.mtime !== mtime) {
			try {
				const { SKILLS } = (await import(`${SKILLS_FILE.href}?v=${mtime}`)) as { SKILLS: readonly Skill[] };
				loaded = { mtime, skills: SKILLS.map(describeSkill), error: null };
			} catch (e) {
				loaded = { mtime, skills: loaded?.skills ?? [], error: `brain/skills.ts did not load: ${(e as Error).message}` };
			}
		}
		return { file: "brain/skills.ts", ...loaded };
	};
}

function describeSkill(k: Skill): LibrarySkill {
	return { name: k.name, says: k.says, amount: k.amount ? { says: k.amount.says, unit: k.amount.unit, levels: [...k.amount.levels] } : null, ready: k.ready ? precondition(k.ready) : null };
}

/** `(f) => f.object_held && !f.at_set_down_height` reads as `object_held && !at_set_down_height`. */
function precondition(ready: NonNullable<Skill["ready"]>): string {
	const src = ready.toString();
	const arrow = /^\(?\s*(\w+)\s*\)?\s*=>\s*([\s\S]*)$/.exec(src);
	return arrow ? arrow[2]!.replace(new RegExp(`\\b${arrow[1]}\\.`, "g"), "").trim() : src;
}

/** Claude's rewrites of the words Jev reads (brain/prompts.ts), or why they could not be read. */
function promptsNow(runsDir: string) {
	const file = join(runsDir, "prompts.json");
	try {
		return { file, ...readPrompts(runsDir) };
	} catch (e) {
		return { file, texts: {}, history: [], error: `${file} did not parse: ${(e as Error).message}` };
	}
}

async function listRuns(store: Store) {
	const root = await store.harness.root(ctx).catch(() => undefined);
	const index = root ? await store.harness.snapshot(RunIndex, root.id, ctx) : undefined;
	const runs = [];
	for (const r of [...(index?.runs ?? [])].reverse()) {
		const run = await store.harness.snapshot(RunDoc, r.conversationId as ConversationId, ctx);
		if (run) runs.push({ id: r.conversationId, kind: run.kind, goal: run.goal, status: run.status, startedAt: run.startedAt, endedAt: run.endedAt, planner: run.planner, decider: run.decider, summary: run.summary });
	}
	return { runs };
}

async function runDetail(store: Store, frames: Recorder, id: ConversationId) {
	const run = await store.harness.snapshot(RunDoc, id, ctx);
	if (!run) return { error: "no such run" };
	const entries = (await runEntries(store, id)).map((e) => ({ id: String(e.id), kind: e.kind, data: e.data, model: e.model ? frames.redact(e.model) : undefined }));
	return { id, run, entries };
}

async function findTick(store: Store, run: ConversationId, entry: string): Promise<TickData | undefined> {
	const hit = (await runEntries(store, run)).find((e) => String(e.id) === entry && e.kind === TickEntry.kind);
	return hit?.data as TickData | undefined;
}

function readBody(req: IncomingMessage): Promise<string> {
	return new Promise((resolve, reject) => {
		let data = "";
		req.on("data", (c) => (data += c));
		req.on("end", () => resolve(data || "{}"));
		req.on("error", reject);
	});
}
