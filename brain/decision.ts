import type { ClassifierAnswer, ClassifierQuestion } from "@earendil-works/pi-ai";
import type { Cam, Frame } from "./arm.ts";
import type { Config, DeciderKind } from "./config.ts";

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type JsonObject = { [key: string]: Json };

export type ImageName = Cam;

export interface BoxQuestion {
	type: "box";
	instructions: string;
}
export interface Box {
	left: number;
	top: number;
	right: number;
	bottom: number;
}
export interface BoxAnswer {
	type: "box";
	box: Box;
	confidence: number;
}
export type Question = ClassifierQuestion | BoxQuestion;
export type Answer = ClassifierAnswer | BoxAnswer;
export type Answers = Record<string, Answer>;

export interface DecisionRequest {
	state: JsonObject;
	images: Partial<Record<ImageName, Frame>>;
	questions: Record<string, Question>;
}

export interface Usage {
	input: number;
	output: number;
}

export interface Exchange {
	request: unknown;
	response: unknown;
	usage: Usage | null;
}

export interface DecideCtx {
	/** tick: a raw move in a task; step: a skill choice toward a goal (brain/director.ts); ask: a look inside a program or skill; replay: a re-ask. */
	role: "tick" | "step" | "ask" | "replay";
	task?: string;
	tick?: number;
	exchange?: (e: Exchange) => void;
}

export interface Caps {
	maxSide: number;
}

export interface DecisionModel {
	id: string;
	caps: Caps;
	decide(req: DecisionRequest, ctx: DecideCtx): Promise<Answers>;
}

export class DecisionError extends Error {}

export function imagePath(name: ImageName): string {
	return `images.${name}`;
}

export function topChoice(a: Answer | undefined): { choice: string; p: number } | undefined {
	if (a?.type !== "choice") return undefined;
	return { choice: a.choice, p: a.probabilities[a.choice] ?? a.confidence };
}

export function yesProbability(a: Answer | undefined): number | undefined {
	return a?.type === "bool" ? a.probability : undefined;
}

export interface DeciderDeps {
	config: Config;
	models?: import("@earendil-works/pi-ai/models").Models;
}

export async function makeDecider(kind: DeciderKind, deps: DeciderDeps): Promise<DecisionModel> {
	const { config } = deps;
	if (kind === "jev" || kind === "h2o") {
		const { systemOne, JEV_BASE_URL, H2O_BASE_URL } = await import("./deciders/systemone.ts");
		const jev = kind === "jev";
		return systemOne({
			id: jev ? "jev" : "h2o",
			baseUrl: config.decider.baseUrl || (jev ? JEV_BASE_URL : H2O_BASE_URL),
			model: config.decider.model || (jev ? "jev-latest" : "h2o-lightning-4b"),
			apiKey: jev ? requireKey("TYPESAFE_API_KEY", "--decider jev") : undefined,
		});
	}
	const { chatDecider } = await import("./deciders/chat.ts");
	if (!deps.models) throw new Error("the chat decider needs a pi-ai Models collection");
	const [provider, ...rest] = config.decider.chat.split("/");
	return chatDecider(deps.models, provider!, rest.join("/"));
}

function requireKey(name: string, why: string): string {
	const key = process.env[name];
	if (!key) throw new Error(`${why} needs ${name} (in .env or the environment)`);
	return key;
}
