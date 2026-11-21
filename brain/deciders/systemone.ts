import { TYPESAFE_CLASSIFIER_MODELS } from "@earendil-works/pi-ai/providers/typesafe.models";
import { DecisionError, type Answer, type Answers, type DecisionModel, type DecisionRequest, type Exchange, type JsonObject, type Question } from "../decision.ts";

export const JEV_BASE_URL = Object.values(TYPESAFE_CLASSIFIER_MODELS).find((m) => m.id === "jev-latest")?.baseUrl ?? "https://api.typesafe.ai/v1/";
export const H2O_BASE_URL = "http://127.0.0.1:8741/v1";

const IMAGE_OPT_IN = { "TypeSafe-Danger-Zone-Images": "true" };

export interface SystemOneOptions {
	id: string;
	baseUrl: string;
	model: string;
	apiKey?: string;
	maxSide?: number;
	timeoutMs?: number;
}

export function systemOne(opts: SystemOneOptions): DecisionModel {
	const url = new URL("systemone", opts.baseUrl.replace(/\/*$/, "/"));
	return {
		id: opts.id,
		caps: { maxSide: opts.maxSide ?? 640 },
		async decide(req, ctx) {
			const request = wireRequest(opts.model, req);
			const headers: Record<string, string> = { "Content-Type": "application/json", ...IMAGE_OPT_IN };
			if (opts.apiKey) headers.Authorization = `Bearer ${opts.apiKey}`;
			let res: Response;
			try {
				res = await fetch(url, {
					method: "POST",
					headers,
					body: JSON.stringify(request),
					signal: AbortSignal.timeout(opts.timeoutMs ?? 20_000),
				});
			} catch (e) {
				ctx.exchange?.({ request, response: { error: (e as Error).message }, usage: null });
				throw new DecisionError(`${opts.id} at ${url.origin} unreachable: ${(e as Error).message}`);
			}
			const text = await res.text();
			const body = parseJson(text);
			const usage = readUsage(body);
			const exchange: Exchange = { request, response: res.ok ? body : { status: res.status, body: body ?? text }, usage };
			ctx.exchange?.(exchange);
			if (!res.ok) throw new DecisionError(`${opts.id} answered HTTP ${res.status}: ${text.slice(0, 400)}`);
			return readAnswers(opts.id, req.questions, body);
		},
	};
}

export function wireRequest(model: string, req: DecisionRequest): JsonObject {
	const refs: Record<string, string> = {};
	const assets: JsonObject = {};
	for (const [name, frame] of Object.entries(req.images)) {
		if (!frame) continue;
		const ref = `${name}-${frame.sha256.slice(0, 16)}`;
		refs[name] = ref;
		assets[ref] = { type: "image", format: "jpeg", data_base64: frame.data_b64 };
	}
	const questions: JsonObject = {};
	for (const [id, q] of Object.entries(req.questions)) questions[id] = wireQuestion(q);
	const request: JsonObject = { model, state: Object.keys(refs).length ? { ...req.state, images: refs } : req.state, questions };
	if (Object.keys(assets).length) request.assets = assets;
	return request;
}

function wireQuestion(q: Question): JsonObject {
	if (q.type === "box") return { type: "box", instructions: q.instructions };
	if (q.type === "bool") return { type: "noul", instructions: q.instructions, criteria: q.criteria };
	if (q.type === "choice") return { type: "choice", instructions: q.instructions, criteria: q.criteria };
	return { type: "score", instructions: q.instructions, criteria: q.criteria };
}

export function readAnswers(label: string, questions: Record<string, Question>, body: unknown): Answers {
	const raw = (body as { answers?: Record<string, any> } | null)?.answers;
	if (!raw || typeof raw !== "object") throw new DecisionError(`${label} returned no answers`);
	const out: Answers = {};
	for (const [id, q] of Object.entries(questions)) {
		const a = raw[id];
		if (!a || typeof a !== "object") throw new DecisionError(`${label} gave no answer to ${id}`);
		out[id] = readAnswer(label, id, q, a);
	}
	return out;
}

function readAnswer(label: string, id: string, q: Question, a: any): Answer {
	const num = (v: unknown, what: string): number => {
		if (typeof v !== "number" || !Number.isFinite(v)) throw new DecisionError(`${label} returned an invalid ${what} for ${id}`);
		return v;
	};
	if (q.type === "choice") {
		if (a.type !== "choice" || typeof a.choice !== "string") throw new DecisionError(`${label} did not answer ${id} with a choice`);
		const probabilities = Object.fromEntries(Object.entries(a.probabilities ?? {}).map(([k, v]) => [k, num(v, `probability of ${k}`)]));
		return { type: "choice", choice: a.choice, probabilities, confidence: num(a.confidence, "confidence") };
	}
	if (q.type === "bool") {
		if (a.type !== "noul") throw new DecisionError(`${label} did not answer ${id} with a noul`);
		return { type: "bool", probability: num(a.noul, "noul") };
	}
	if (q.type === "box") {
		if (a.type !== "box" || !a.box) throw new DecisionError(`${label} did not answer ${id} with a box`);
		const box = { left: num(a.box.left, "box.left"), top: num(a.box.top, "box.top"), right: num(a.box.right, "box.right"), bottom: num(a.box.bottom, "box.bottom") };
		return { type: "box", box, confidence: num(a.confidence, "confidence") };
	}
	if (a.type !== "score") throw new DecisionError(`${label} did not answer ${id} with a score`);
	return { type: "score", score: num(a.score, "score"), confidence: num(a.confidence, "confidence") };
}

function readUsage(body: unknown): Exchange["usage"] {
	const u = (body as { usage?: { input_tokens?: number; output_tokens?: number } } | null)?.usage;
	if (!u) return null;
	return { input: u.input_tokens ?? 0, output: u.output_tokens ?? 0 };
}

function parseJson(text: string): unknown {
	try {
		return JSON.parse(text);
	} catch {
		return null;
	}
}
