import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { defineExtension, defineTool, GenerationTask, hook, section, type ConversationId, type Extension } from "@earendil-works/pi-durable";
import type { Frame } from "./arm.ts";
import { savePromptEdits } from "./prompts.ts";
import { conversationSink, ctx, Recorder, RunDoc, updateRun, type RunState, type Store } from "./record.ts";
import type { Skill } from "./skills.ts";
import { WRIST_CONVENTION } from "./task.ts";

/** States where code stops and asks Claude, whatever Jev chose: the claims a wrong step would build on. */
export type CheckpointKind = "found" | "aligned" | "gripped" | "done" | "handback";

export const CLAIMS: Record<CheckpointKind, string> = {
	found: "find_object says it has located the object, and the arm is about to move above it. Is the object in the pictures, and is it the object named?",
	aligned:
		"The open jaws are down at gripping height, the camera almost straight above the jaw tips. Closing grips only if the object stands alongside the fixed jaw, not at or past its tip: the centre of its top must be clearly below the fixed jaw's tip in the picture (toward the bottom edge by at least a third of the top's width; cut off by the bottom edge is fine), with the jaw's inner edge running beside the object below the tip. A gap up to the top's width between jaw and object is fine; the moving jaw sweeps it in. If the top's centre is level with or above the tip, or the object touches the jaw only at its tip, it is not aligned: move the jaws out by at least half the top's width. Too far out is much less harmful than too little.",
	gripped: "The jaws closed on something and lift says the object rides in the jaws. Is the object held between the two jaws, off the desk?",
	done: "Jev says the goal is accomplished. Is it, from the pictures?",
	handback: "Jev stopped before the goal was met. What went wrong, which skill would get it moving again, and would clearer words for Jev have helped?",
};

/** What code does when Claude disagrees, so Claude knows what its answer sets off. */
export const ON_DISAGREE: Record<CheckpointKind, string> = {
	found: "the object counts as not found, so Jev will look again",
	aligned: "code moves the jaws by your move (sideways moves rise above the object first, so the jaws don't shove it) and asks you again",
	gripped: "the grip counts as missed, so the jaws open and Jev tries again",
	done: "the goal counts as not done, so Jev keeps going",
	handback: "the goal stops here unless you name next_skill, which code then runs (a few times a goal at most); your edits apply from the next step",
};

export interface Look {
	kind: CheckpointKind;
	goal: string;
	what: string;
	/** Why Jev stopped, for a handback. */
	reason?: string;
	facts: object;
	recent: readonly object[];
	texts: Record<string, string>;
	/** The skills whose preconditions hold now: the ones Jev could be offered, or code could run as next_skill. */
	ready?: string[];
	frames: { label: string; frame: Frame }[];
}

export interface Verdict {
	agree: boolean;
	why: string;
	/** At a handback: the skill code should run next instead of stopping. */
	next?: { skill: string; amount?: number };
	/** At an aligned check: how far the jaws should move to bring the object into their opening. */
	move?: { right_cm: number; out_cm: number };
	edited: string[];
	refused: string[];
}

export interface Overseer {
	look(l: Look): Promise<Verdict>;
}

export interface OverseerServices {
	store: Store;
	skills: readonly Skill[];
}

const pending = new Map<ConversationId, { kind: CheckpointKind; n: number; frames: string[] }>();

export function overseerExtension(svc: OverseerServices): Extension {
	const recorderFor = (id: ConversationId) => new Recorder(svc.store.framesDir, conversationSink(svc.store, id));
	return defineExtension({
		name: "valet-overseer",
		sections: [section("preamble", () => PREAMBLE, { tag: false })],
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
				name: "verdict",
				description: "Your judgement of this checkpoint's claim, and any rewrites of the words Jev reads. Call it once per checkpoint.",
				parameters: {
					type: "object",
					properties: {
						agree: { type: "boolean", description: "whether the pictures bear the claim out" },
						why: { type: "string", description: "one or two sentences on what the pictures show" },
						move: {
							type: "object",
							description: "at an aligned check you disagree with: how far to move the jaws, in cm, to bring the object into their opening",
							properties: {
								right_cm: { type: "number", description: "toward the arm's right (image right); negative for left" },
								out_cm: { type: "number", description: "away from the base (image up); negative toward the base" },
							},
							required: ["right_cm", "out_cm"],
							additionalProperties: false,
						},
						next_skill: { type: "string", description: "at a handback only: the skill to run next instead of stopping, when the pictures show what would get the goal moving again; leave out to stop" },
						next_amount: { type: "number", description: "the amount for next_skill, in its unit, when it has one" },
						edits: {
							type: "array",
							description: "rewrites of the words Jev reads, only where a picture shows Jev misjudging something clearer words would fix",
							items: {
								type: "object",
								properties: { target: { type: "string" }, text: { type: "string" }, reason: { type: "string" } },
								required: ["target", "text", "reason"],
								additionalProperties: false,
							},
						},
					},
					required: ["agree", "why"],
					additionalProperties: false,
				},
				execute: async (args, api) => {
					const { agree, why, edits = [], next_skill, next_amount, move } = args as { agree: boolean; why: string; move?: { right_cm: number; out_cm: number }; next_skill?: string; next_amount?: number; edits?: { target: string; text: string; reason: string }[] };
					const at = pending.get(api.conversationId);
					const refused = savePromptEdits(svc.store.runsDir, svc.skills, edits.map((e) => ({ ...e, run: api.conversationId as number, ...(at ? { checkpoint: at.kind } : {}) })));
					const edited = edits.map((e) => e.target).filter((t) => !refused.some((r) => r.startsWith(`${t}:`)));
					await updateRun(svc.store, api.conversationId, (run) => {
						(run.checkpoints ??= []).push({ n: at?.n ?? 0, kind: at?.kind ?? "unknown", agree, why, ...(move ? { move } : {}), ...(next_skill ? { next: { skill: next_skill, ...(next_amount !== undefined ? { amount: next_amount } : {}) } } : {}), edits: edits as never, edited, refused, frames: at?.frames ?? [], at: Date.now() });
					});
					const text = refused.length ? `Recorded. Not applied: ${refused.join("; ")}` : "Recorded.";
					return { output: [{ type: "text", text } as TextContent], control: { terminate: true } };
				},
			}),
		],
	});
}

export async function startOverseer(store: Store, id: ConversationId, model: { provider: string; modelId: string }, recorder: Recorder): Promise<Overseer> {
	const conv = (await store.harness.conversation(id, ctx))!;
	await conv.configure({ model, maxTurns: 3 }, ctx);
	let n = 0;
	return {
		async look(l) {
			n++;
			const frames = l.frames.map((f) => recorder.saveFrame(f.frame).sha256);
			pending.set(id, { kind: l.kind, n, frames });
			const text = JSON.stringify(
				{
					checkpoint: l.kind,
					claim: CLAIMS[l.kind],
					if_you_disagree: ON_DISAGREE[l.kind],
					...(l.reason ? { why_jev_stopped: l.reason } : {}),
					goal: l.goal,
					object: l.what,
					facts: l.facts,
					recent_steps: l.recent,
					words_jev_reads: l.texts,
					...(l.ready ? { skills_ready_now: l.ready } : {}),
					pictures: l.frames.map((f) => f.label),
					camera: WRIST_CONVENTION,
				},
				null,
				1,
			);
			const images: ImageContent[] = l.frames.map((f) => ({ type: "image", data: f.frame.data_b64, mimeType: f.frame.mime }));
			const sub = await conv.submit({ type: "input", content: [{ type: "text", text }, ...images], requestId: `checkpoint-${n}` }, ctx);
			await sub.wait(ctx);
			const run = (await store.harness.snapshot(RunDoc, id, ctx)) as RunState;
			const mine = ((run.checkpoints ?? []) as { n: number; agree: boolean; why: string; next?: { skill: string; amount?: number }; move?: { right_cm: number; out_cm: number }; edited: string[]; refused: string[] }[]).filter((c) => c.n === n).at(-1);
			pending.delete(id);
			return mine ? { agree: mine.agree, why: mine.why, ...(mine.next ? { next: mine.next } : {}), ...(mine.move ? { move: mine.move } : {}), edited: mine.edited, refused: mine.refused } : { agree: true, why: "Claude gave no verdict; going on as Jev said", edited: [], refused: [] };
		},
	};
}

const PREAMBLE = `You oversee a small robot arm, an SO-101 with a two-jaw gripper and a camera on the wrist. Jev, a fast vision \
model, picks each step from a fixed set of code skills by reading short texts and the wrist picture; code carries the \
step out. At checkpoints code stops and shows you pictures and a claim; judge the claim from the pictures and call \
verdict once. Disagree only when the pictures show the claim is wrong: code acts on your answer as the checkpoint says. \
At a handback, when one of skills_ready_now would get the goal moving again, name it in next_skill (with next_amount \
for a skill with an amount): code runs it and Jev carries on from there.

You also own the words Jev reads (words_jev_reads): the question asking for the next skill (target "choice"), each \
skill's description (target: its name) and each how-far question (target "<skill>.amount"). When the pictures and \
recent steps show Jev misjudging something that clearer words would fix, rewrite that target in verdict's edits; edits \
apply from the next step and in later runs. Keep each text to one or two short sentences about what can be seen in the \
wrist picture, and do not restate facts code already gives Jev. Leave the words alone when they are not the problem.`;
