import { createServer, type IncomingHttpHeaders } from "node:http";
import type { AddressInfo } from "node:net";
import { ACTIONS, MOVES, type Action } from "../task.ts";

export interface SystemOneStub {
	baseUrl: string;
	requests: { path: string; headers: IncomingHttpHeaders; body: any }[];
	respond(status: number, body: unknown): void;
	close(): Promise<void>;
}

export async function startSystemOneStub(): Promise<SystemOneStub> {
	const requests: SystemOneStub["requests"] = [];
	let reply: { status: number; body: unknown } = { status: 500, body: { error: "nothing scripted" } };
	const server = createServer((req, res) => {
		let data = "";
		req.on("data", (c) => (data += c));
		req.on("end", () => {
			requests.push({ path: req.url ?? "", headers: req.headers, body: JSON.parse(data || "{}") });
			res.writeHead(reply.status, { "Content-Type": "application/json" });
			res.end(JSON.stringify(reply.body));
		});
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	return {
		baseUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`,
		requests,
		respond: (status, body) => (reply = { status, body }),
		close: () =>
			new Promise<void>((resolve) => {
				server.close(() => resolve());
				server.closeAllConnections();
			}),
	};
}

export function jevAnswers(choice: Action = "reach_out") {
	return {
		model: "jev-1.14.0",
		answers: {
			act: { type: "choice", choice, probabilities: Object.fromEntries(ACTIONS.map((a) => [a, a === choice ? 0.8 : 0.2 / (ACTIONS.length - 1)])), confidence: 0.8 },
			...Object.fromEntries(Object.keys(MOVES).map((m) => [`far_${m}`, { type: "score", score: 1.4, confidence: 0.5, probabilities: { "0": 0.1, "1": 0.4, "2": 0.5, "3": 0, "4": 0 } }])),
		},
		usage: { input_tokens: 1290, output_tokens: 21 },
	};
}
