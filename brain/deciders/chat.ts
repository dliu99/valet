import type { Context, Model, Api } from "@earendil-works/pi-ai";
import type { Models } from "@earendil-works/pi-ai/models";
import { DecisionError, imagePath, type Answer, type Answers, type DecisionModel, type ImageName, type Question } from "../decision.ts";

const SYSTEM = `You answer structured questions about a robot arm from its state and camera images, the way a calibrated \
decision model would: for each question give probabilities, not explanations. Reply with one JSON object and nothing else.`;

const FORMAT = `Reply with one JSON object whose keys are the question ids:
- a choice question: {"probabilities": {"<option id>": p, ...}} over every option, summing to 1;
- a yes/no question: {"probability": p}, the probability that the answer is yes (true);
- a score question: {"score": s, "confidence": c}, s the position on its levels, numbered from 0 in the order given, and may \
fall between two levels;
- a box question: {"box": {"left": l, "top": t, "right": r, "bottom": b}, "confidence": c}, edges as fractions 0..1 of the image from its \
top-left corner.
`;

export function chatDecider(models: Models, provider: string, modelId: string): DecisionModel {
	const model = models.getModel(provider, modelId) as Model<Api> | undefined;
	if (!model) throw new Error(`no pi-ai chat model ${provider}/${modelId}`);
	if (!model.input.includes("image")) throw new Error(`${provider}/${modelId} does not take images`);
	return {
		id: `chat:${provider}/${modelId}`,
		caps: { maxSide: 1024 },
		async decide(req, ctx) {
			const names = Object.keys(req.images) as ImageName[];
			const content: Extract<Context["messages"][number], { role: "user" }>["content"] = [
				{ type: "text", text: `State:\n${JSON.stringify({ ...req.state, images: Object.fromEntries(names.map((n) => [n, `the image labelled ${imagePath(n)}`])) }, null, 2)}` },
			];
			for (const name of names) {
				const frame = req.images[name]!;
				content.push({ type: "text", text: `Image \`${imagePath(name)}\` (${frame.width}x${frame.height}):` });
				content.push({ type: "image", data: frame.data_b64, mimeType: frame.mime });
			}
			content.push({ type: "text", text: `Questions:\n${JSON.stringify(describe(req.questions), null, 2)}\n\n${FORMAT}` });
			let request: unknown = null;
			const reply = await models.complete(model, { systemPrompt: SYSTEM, messages: [{ role: "user", content, timestamp: Date.now() }] }, {
				onPayload: (payload: unknown) => {
					request = payload;
					return undefined;
				},
			} as never);
			const usage = reply.usage ? { input: reply.usage.input, output: reply.usage.output } : null;
			ctx.exchange?.({ request, response: reply, usage });
			if (reply.stopReason === "error" || reply.stopReason === "aborted") throw new DecisionError(`${provider}/${modelId}: ${reply.errorMessage ?? reply.stopReason}`);
			const text = reply.content.flatMap((c) => (c.type === "text" ? [c.text] : [])).join("");
			return readChatAnswers(req.questions, text);
		},
	};
}

function describe(questions: Record<string, Question>) {
	return Object.fromEntries(
		Object.entries(questions).map(([id, q]) => [
			id,
			q.type === "bool" ? { type: "yes/no", instructions: q.instructions, criteria: q.criteria } : q,
		]),
	);
}

export function readChatAnswers(questions: Record<string, Question>, text: string): Answers {
	const start = text.indexOf("{");
	const end = text.lastIndexOf("}");
	let raw: Record<string, any>;
	try {
		raw = JSON.parse(text.slice(start, end + 1));
	} catch {
		throw new DecisionError(`the chat model's reply is not JSON: ${text.slice(0, 200)}`);
	}
	const out: Answers = {};
	for (const [id, q] of Object.entries(questions)) {
		const a = raw[id];
		if (!a) throw new DecisionError(`the chat model gave no answer to ${id}`);
		out[id] = readOne(id, q, a);
	}
	return out;
}

function readOne(id: string, q: Question, a: any): Answer {
	const p = (v: unknown) => Math.min(1, Math.max(0, Number(v) || 0));
	if (q.type === "choice") {
		const keys = Object.keys(q.criteria);
		const weights = keys.map((k) => p(a.probabilities?.[k]));
		const total = weights.reduce((s, w) => s + w, 0) || 1;
		const probabilities = Object.fromEntries(keys.map((k, i) => [k, weights[i]! / total]));
		const choice = keys.reduce((best, k) => (probabilities[k]! > probabilities[best]! ? k : best), keys[0]!);
		return { type: "choice", choice, probabilities, confidence: probabilities[choice]! };
	}
	if (q.type === "bool") return { type: "bool", probability: p(a.probability) };
	if (q.type === "box") {
		if (!a.box) throw new DecisionError(`the chat model gave no box for ${id}`);
		return { type: "box", box: { left: p(a.box.left), top: p(a.box.top), right: p(a.box.right), bottom: p(a.box.bottom) }, confidence: p(a.confidence) };
	}
	return { type: "score", score: Math.min(q.criteria.length - 1, Math.max(0, Number(a.score) || 0)), confidence: p(a.confidence) };
}
