#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { createInterface } from "node:readline/promises";
import { parseArgs } from "node:util";
import type { MutableModels } from "@earendil-works/pi-ai/models";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import { createRegistry, type ConversationId } from "@earendil-works/pi-durable";
import { armClient, type ArmClient } from "./arm.ts";
import { DECIDER_KINDS, loadConfig, loadEnv, type Config, type DeciderKind } from "./config.ts";
import { makeDecider } from "./decision.ts";
import { runGoal, runSkill, type Exit } from "./director.ts";
import { overseerExtension, startOverseer } from "./overseer.ts";
import { plannerExtension, planRun, type PlannerServices } from "./planner.ts";
import type { ProgramDeps } from "./program.ts";
import { readPrompts } from "./prompts.ts";
import { addTask, conversationSink, ctx, openStore, recorded, recordedTicks, Recorder, startRun, StepEntry, TickEntry, updateRun, type RunStatus, type StepData, type Store, type TickData } from "./record.ts";
import { skillNamed, SKILLS, type Scratch } from "./skills.ts";
import { askTask, reask } from "./stage.ts";
import { validate, type Task } from "./task.ts";
import { serveUI } from "./ui.ts";

const HELP = `valet: Jev picks every move of the SO-101, Claude writes Jev's cards, armd owns the hardware.

  valet run "<goal>"        plan and carry out a goal on the arm
  valet do "<goal>" --what "<object>" [--steps N] [--no-claude]
                            Jev picks skills (and how far) toward a goal on the arm, at most N steps (default 24)
  valet skill <name> --what "<object>" [--amount N]
                            run one skill on the arm (pick, place_left, place_right, place_back, ...)
  valet ask <task.json>     one decision on the current frames, no motion
  valet replay --run <id> --task <task.json> --decider <kind>
                            re-ask a run's recorded ticks with a task; report agreement with the recorded actions
  valet ui                  the debug UI over runs/valet.sqlite, read-only

  --decider <kind>          jev | h2o | chat (default from [decider], jev)
  --planner <p/m>           any pi-ai chat model, e.g. anthropic/claude-opus-5-5
  --armd <url>              armd's URL (default from [armd], http://127.0.0.1:8750)
  --config <file>           valet.toml (default: ./valet.toml if present)
  --runs <dir>              where runs are kept (default runs)
  --port <n>                the debug UI's port (default 8760)
  --ui                      with run or do: serve the debug UI while the run goes

Keys (TYPESAFE_API_KEY, ANTHROPIC_API_KEY) come from .env here, else from the main checkout's .env.
`;

async function main(): Promise<number> {
	const { positionals, values } = parseArgs({
		allowPositionals: true,
		options: {
			decider: { type: "string" },
			planner: { type: "string" },
			armd: { type: "string" },
			config: { type: "string" },
			runs: { type: "string" },
			port: { type: "string" },
			run: { type: "string" },
			task: { type: "string" },
			what: { type: "string" },
			steps: { type: "string" },
			amount: { type: "string" },
			ui: { type: "boolean" },
			"no-claude": { type: "boolean" },
			help: { type: "boolean", short: "h" },
		},
	});
	const [cmd, arg] = positionals;
	if (values.help || !cmd || !["run", "do", "skill", "ask", "replay", "ui"].includes(cmd)) {
		process.stdout.write(HELP);
		return cmd && !values.help ? 2 : 0;
	}
	loadEnv();
	const config = loadConfig(values.config);
	if (values.runs) config.runsDir = values.runs;
	if (values.port) config.uiPort = Number(values.port);
	if (values.decider) {
		if (!DECIDER_KINDS.includes(values.decider as DeciderKind)) throw new Error(`--decider must be one of ${DECIDER_KINDS.join(", ")}`);
		config.decider.kind = values.decider as DeciderKind;
	}
	if (values.armd) config.armdUrl = values.armd;

	if (cmd === "ui") {
		const store = await openStore(config.runsDir, { readOnly: true });
		const models = builtinModels();
		await serveUI({ store, config, models, port: config.uiPort });
		console.log(`debug UI (read-only) on http://127.0.0.1:${config.uiPort}`);
		return new Promise(() => {});
	}

	const arm = armClient(config.armdUrl);
	const models = builtinModels();
	if (cmd === "run") return run(arg, config, arm, models, values.planner ?? `${config.planner.provider}/${config.planner.model}`, Boolean(values.ui));
	if (cmd === "do") return doGoal(arg, values.what, values.steps, config, arm, models, Boolean(values.ui), values["no-claude"] ? null : splitModel(values.planner ?? `${config.planner.provider}/${config.planner.model}`));
	if (cmd === "skill") return oneSkill(arg, values.what, values.amount, config, arm, models);
	if (cmd === "ask") return ask(arg, config, arm, models);
	return replay(values.run, values.task, config, models);
}

async function run(goal: string | undefined, config: Config, arm: ArmClient, models: MutableModels, plannerSpec: string, withUI: boolean): Promise<number> {
	if (!goal) throw new Error('give the goal: valet run "<goal>"');
	const planner = splitModel(plannerSpec);
	if (!models.getModel(planner.provider, planner.modelId)) throw new Error(`no pi-ai chat model ${plannerSpec}`);
	const decider = await makeDecider(config.decider.kind, { config, models });
	const registry = createRegistry();
	const store = await openStore(config.runsDir, { models, registry });
	const svc: PlannerServices = { config, arm, decider, store, person: askPerson };
	registry.install(plannerExtension(svc));
	const quit = async (code: number) => {
		await store.close();
		process.exit(code);
	};
	process.on("SIGINT", () => void quit(130));
	await stopUnfinishedRuns(store);
	if (withUI) {
		await serveUI({ store, config, models, port: config.uiPort });
		console.log(`debug UI on http://127.0.0.1:${config.uiPort}`);
	}
	console.log(`goal: ${goal}\nplanner ${planner.provider}/${planner.modelId}, decider ${decider.id}, armd ${arm.url}`);
	const { id, run: result } = await planRun(svc, goal, planner, (runId) => void follow(store, runId));
	console.log(`run #${id}: ${result.status}: ${result.summary ?? ""}`);
	await store.close();
	return result.status === "done" ? 0 : 1;
}

async function doGoal(text: string | undefined, what: string | undefined, stepsArg: string | undefined, config: Config, arm: ArmClient, models: MutableModels, withUI: boolean, claude: { provider: string; modelId: string } | null): Promise<number> {
	if (!text || !what) throw new Error('valet do "<goal>" --what "<object>" [--steps N]');
	const steps = stepsArg === undefined ? undefined : Number(stepsArg);
	if (steps !== undefined && !(Number.isInteger(steps) && steps > 0)) throw new Error("--steps is a whole number above 0");
	if (claude && !models.getModel(claude.provider, claude.modelId)) throw new Error(`no pi-ai chat model ${claude.provider}/${claude.modelId}`);
	const decider = await makeDecider(config.decider.kind, { config, models });
	const registry = createRegistry();
	const store = await openStore(config.runsDir, { models, registry });
	registry.install(overseerExtension({ store, skills: SKILLS }));
	try {
		await stopUnfinishedRuns(store);
		const id = await startRun(store, { kind: "do", goal: `${text} (${what})`, planner: claude ? `${claude.provider}/${claude.modelId}` : "none", decider: decider.id });
		if (withUI) {
			await serveUI({ store, config, models, port: config.uiPort });
			console.log(`debug UI on http://127.0.0.1:${config.uiPort}`);
		}
		console.log(`run #${id}: ${text}\nobject ${what}, decider ${decider.id}, armd ${arm.url}`);
		void follow(store, id);
		const recorder = new Recorder(store.framesDir, conversationSink(store, id));
		const overseer = claude ? await startOverseer(store, id, claude, recorder) : undefined;
		const helpers = { prompts: () => readPrompts(config.runsDir), ...(overseer ? { overseer } : {}) };
		const report = await runGoal({ text, what, ...(steps ? { steps } : {}) }, { arm, model: recorded(decider, recorder), recorder, view: config.viewpoints.wrist }, stopOnCtrlC(), SKILLS, {}, helpers);
		await updateRun(store, id, (r) => Object.assign(r, { status: RUN_STATUS[report.exit], endedAt: Date.now(), summary: `${report.exit}: ${report.reason}` }));
		console.log(`run #${id}: ${report.exit}: ${report.reason}`);
		return report.exit === "done" ? 0 : 1;
	} finally {
		await store.close();
	}
}

const RUN_STATUS: Record<Exit, RunStatus> = { done: "done", stuck: "stopped", unsure: "stopped", budget: "stopped", interrupted: "stopped", failed: "failed" };

async function oneSkill(name: string | undefined, what: string | undefined, amountArg: string | undefined, config: Config, arm: ArmClient, models: MutableModels): Promise<number> {
	if (!name || !what) throw new Error('valet skill <name> --what "<object>" [--amount N]');
	const skill = skillNamed(name);
	const amount = amountArg === undefined ? undefined : Number(amountArg);
	if (amount !== undefined && (!skill.amount || !Number.isFinite(amount))) throw new Error(skill.amount ? "--amount is a number" : `${name} takes no amount`);
	const decider = await makeDecider(config.decider.kind, { config, models });
	const store = await openStore(config.runsDir, { models });
	try {
		await stopUnfinishedRuns(store);
		await arm.hold();
		const label = `${name}${skill.amount && amount !== undefined ? ` ${amount} ${skill.amount.unit}` : ""}: ${what}`;
		const id = await startRun(store, { kind: "skill", goal: label, planner: "none", decider: decider.id });
		const recorder = new Recorder(store.framesDir, conversationSink(store, id));
		const deps: ProgramDeps = { arm, model: recorded(decider, recorder), recorder, view: config.viewpoints.wrist };
		const scratch: Scratch = {};
		const args = { what, ...(amount !== undefined ? { amount } : {}) };
		const run = await runSkill(skill, args, scratch, deps, stopOnCtrlC(), "skill");
		const step: StepData = { task: "skill", step: 1, goal: label, what, skill: name, amount: amount ?? null, unit: (amount !== undefined && skill.amount?.unit) || null, p: null, ok: run.result.ok, why: run.result.why, result: recorder.redact(run.result), log: run.log, held: recorder.redact(scratch.held ?? null), ms: run.ms };
		await recorder.step(step);
		await updateRun(store, id, (r) => Object.assign(r, { status: run.result.ok ? "done" : "failed", endedAt: Date.now(), summary: run.result.why }));
		console.log(JSON.stringify({ run: id, ...run }, null, 2));
		return run.result.ok ? 0 : 1;
	} finally {
		await store.close();
	}
}

/** The first Ctrl-C stops the goal or skill at its next arm call and holds the arm; a second one quits at once. */
function stopOnCtrlC(): AbortSignal {
	const stop = new AbortController();
	process.on("SIGINT", () => {
		if (stop.signal.aborted) process.exit(130);
		console.log("\nstopping at the next arm call; Ctrl-C again to quit now");
		stop.abort(new Error("stopped from the terminal"));
	});
	return stop.signal;
}

/**
 * Runs an earlier process left unfinished would resume as soon as this process schedules work, and could move the arm
 * for a goal nobody is watching; they are stopped instead, and their records kept.
 */
async function stopUnfinishedRuns(store: Store): Promise<void> {
	const inspection = await store.harness.inspect(ctx);
	const ids = new Set(inspection.tasks.map((t) => t.record.conversationId as ConversationId));
	for (const id of ids) {
		const conv = await store.harness.conversation(id, ctx);
		if (!conv) continue;
		console.log(`stopping run #${id}, left unfinished by an earlier process`);
		await conv.abort(ctx);
		await updateRun(store, id, (r) => {
			if (r.status === "running") Object.assign(r, { status: "stopped", summary: "left unfinished by an earlier process; stopped when a new run started", endedAt: Date.now() });
		}).catch(() => undefined);
	}
}

async function follow(store: Store, id: ConversationId): Promise<void> {
	const conv = await store.harness.conversation(id, ctx);
	if (!conv) return;
	const watch = await conv.watch(ctx);
	const seen = new Set<string>();
	const show = (entries: readonly { id: unknown; kind: string; data?: unknown; model?: readonly unknown[] }[]) => {
		for (const e of entries) {
			const key = String(e.id);
			if (seen.has(key)) continue;
			seen.add(key);
			if (e.kind === TickEntry.kind) {
				const t = e.data as TickData;
				const r = t.reading as { action?: string; p?: number } | null;
				const delta = t.motion && (t.motion as { kind?: string }).kind === "delta" ? ` ${JSON.stringify(t.motion).replace(/"kind":"delta",?/, "")}` : "";
				const what = r?.action ?? Object.keys((t.questions as object | null) ?? {}).join(", ");
				const line = t.error ? `error: ${t.error}` : `${what}${delta || (t.step ? ` ${t.step}` : "")}${r?.p == null ? "" : ` (p ${r.p.toFixed(2)})`}`;
				console.log(`  ${t.role} ${t.task}#${t.tick ?? "-"} ${line}  [${t.ms.model} ms]`);
			} else if (e.kind === StepEntry.kind) {
				const s = e.data as StepData;
				console.log(`  -> ${s.skill}${s.amount == null ? "" : ` ${s.amount} ${s.unit}`}: ${s.ok ? "ok" : "not ok"}: ${s.why}  [${(s.ms / 1000).toFixed(1)} s]`);
			} else if (e.kind === "pi.assistant") {
				const msg = e.model?.[0] as { content?: { type: string; name?: string; text?: string }[] } | undefined;
				for (const c of msg?.content ?? []) {
					if (c.type === "toolCall") console.log(`planner: ${c.name}`);
					else if (c.type === "text" && c.text?.trim()) console.log(`planner: ${c.text.trim().slice(0, 400)}`);
				}
			}
		}
	};
	show(watch.value.entries as never);
	watch.start(async (call) => {
		if (call.kind === "value") show(call.value.entries as never);
	});
}

async function ask(file: string | undefined, config: Config, arm: ArmClient, models: MutableModels): Promise<number> {
	const task = readTask(file);
	const decider = await makeDecider(config.decider.kind, { config, models });
	const store = await openStore(config.runsDir, { models });
	try {
		const id = await startRun(store, { kind: "ask", goal: `ask: ${task.goal}`, planner: "none", decider: decider.id });
		await addTask(store, id, task);
		const recorder = new Recorder(store.framesDir, conversationSink(store, id));
		const result = await askTask(task, { arm, model: recorded(decider, recorder), recorder, view: config.viewpoints.wrist });
		await updateRun(store, id, (r) => Object.assign(r, { status: "done", endedAt: Date.now(), summary: result.reading.action }));
		console.log(JSON.stringify({ run: id, reading: result.reading, would_move: result.motion, answers: result.answers }, null, 2));
		return 0;
	} finally {
		await store.close();
	}
}

async function replay(runId: string | undefined, file: string | undefined, config: Config, models: MutableModels): Promise<number> {
	if (!runId || !file) throw new Error("valet replay --run <id> --task <task.json> --decider <kind>");
	const task = readTask(file);
	const decider = await makeDecider(config.decider.kind, { config, models });
	const store = await openStore(config.runsDir, { models });
	try {
		const ticks = (await recordedTicks(store, Number(runId) as ConversationId)).filter((t) => t.answers);
		if (!ticks.length) throw new Error(`run ${runId} has no recorded ticks`);
		const id = await startRun(store, { kind: "replay", goal: `replay on run ${runId}: ${task.goal}`, planner: "none", decider: decider.id });
		await addTask(store, id, task);
		const recorder = new Recorder(store.framesDir, conversationSink(store, id));
		const model = recorded(decider, recorder);
		let same = 0;
		for (const t of ticks) {
			const { reading } = await reask(task, t, { model, recorder, view: config.viewpoints.wrist });
			const then = (t.reading as { action?: string } | null)?.action;
			if (reading.action === then) same++;
			console.log(`#${t.task}.${t.tick} recorded ${then ?? "?"} -> now ${reading.action} (p ${reading.p.toFixed(2)})`);
		}
		const summary = `same action as recorded on ${same}/${ticks.length} ticks`;
		await updateRun(store, id, (r) => Object.assign(r, { status: "done", endedAt: Date.now(), summary }));
		console.log(summary);
		return 0;
	} finally {
		await store.close();
	}
}

function readTask(file: string | undefined): Task {
	if (!file) throw new Error("give a task file: valet ask <task.json> with {goal, notes?, seconds?}");
	const parsed = JSON.parse(readFileSync(file, "utf8")) as Partial<Task>;
	const task = { id: "0", seconds: 60, ...parsed } as Task;
	const errors = validate(task);
	if (errors.length) throw new Error(`invalid task:\n  ${errors.join("\n  ")}`);
	return task;
}

function splitModel(spec: string): { provider: string; modelId: string } {
	const i = spec.indexOf("/");
	if (i < 1) throw new Error(`expected provider/model, got ${JSON.stringify(spec)}`);
	return { provider: spec.slice(0, i), modelId: spec.slice(i + 1) };
}

async function askPerson(question: string): Promise<string> {
	if (!process.stdin.isTTY) return "No person is at the terminal.";
	const rl = createInterface({ input: process.stdin, output: process.stdout });
	try {
		return await rl.question(`\nThe planner asks: ${question}\n> `);
	} finally {
		rl.close();
	}
}

main().then(
	(code) => process.exit(code),
	(e) => {
		console.error(`valet: ${(e as Error).message}`);
		process.exit(1);
	},
);
