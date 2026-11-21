import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { armClient, type ArmClient } from "../arm.ts";
import type { Answers, DecisionModel, DecisionRequest, JsonObject } from "../decision.ts";
import { recorded, Recorder, type Sink } from "../record.ts";
import type { TaskDeps } from "../stage.ts";
import { startArmdDouble, type ArmdDouble, type DoubleOptions } from "./armd-double.ts";

export const tempDir = (prefix = "valet-test-") => mkdtempSync(join(tmpdir(), prefix));

export interface Recorded {
	kind: string;
	data: JsonObject;
	actsBefore: number;
}

export function memorySink(armd?: ArmdDouble): Sink & { records: Recorded[] } {
	const records: Recorded[] = [];
	return {
		records,
		async append(kind, data) {
			records.push({ kind, data: structuredClone(data), actsBefore: armd?.calls.filter((c) => c === "POST /act").length ?? 0 });
		},
	};
}

export function choose(req: DecisionRequest, action: string, pick: { p?: number; far?: number } = {}): Answers {
	const answers: Answers = {};
	for (const [id, q] of Object.entries(req.questions)) {
		if (q.type === "choice") {
			const options = Object.keys(q.criteria);
			const p = pick.p ?? 0.9;
			const rest = (1 - p) / (options.length - 1);
			answers[id] = { type: "choice", choice: action, probabilities: Object.fromEntries(options.map((o) => [o, o === action ? p : rest])), confidence: p };
		} else if (q.type === "score") answers[id] = { type: "score", score: pick.far ?? 1, confidence: 0.6 };
		else if (q.type === "box") answers[id] = { type: "box", box: { left: 0.4, top: 0.3, right: 0.6, bottom: 0.7 }, confidence: 0.8 };
		else answers[id] = { type: "bool", probability: 0.5 };
	}
	return answers;
}

export function scriptedDecider(reply: (req: DecisionRequest, tick: number | undefined) => Answers | Promise<Answers>): DecisionModel {
	return {
		id: "scripted",
		caps: { maxSide: 640 },
		async decide(req, ctx) {
			const answers = await reply(req, ctx.tick);
			ctx.exchange?.({ request: { questions: Object.keys(req.questions) }, response: answers, usage: null });
			return answers;
		},
	};
}

export interface Rig {
	armd: ArmdDouble;
	arm: ArmClient;
	sink: ReturnType<typeof memorySink>;
	recorder: Recorder;
	deps: TaskDeps;
	close(): Promise<void>;
}

export async function rig(opts: DoubleOptions & { decider?: (arm: ArmClient) => DecisionModel } = {}): Promise<Rig> {
	const armd = await startArmdDouble(opts);
	const arm = armClient(armd.url);
	armd.calls.length = 0;
	const sink = memorySink(armd);
	const recorder = new Recorder(join(tempDir(), "frames"), sink);
	const model = recorded((opts.decider ?? (() => scriptedDecider((req) => choose(req, "stuck"))))(arm), recorder);
	return { armd, arm, sink, recorder, deps: { arm, model, recorder, view: "on the gripper" }, close: armd.close };
}
