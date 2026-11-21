import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { parse } from "smol-toml";

export type DeciderKind = "jev" | "h2o" | "chat";
export const DECIDER_KINDS: readonly DeciderKind[] = ["jev", "h2o", "chat"];

export interface Config {
	armdUrl: string;
	viewpoints: { wrist: string; side: string };
	decider: { kind: DeciderKind; baseUrl: string; model: string; chat: string };
	planner: { provider: string; model: string };
	uiPort: number;
	runsDir: string;
}

export const DEFAULTS: Config = {
	armdUrl: "http://127.0.0.1:8750",
	viewpoints: { wrist: "on the gripper, looking along the jaws", side: "fixed, beside the table" },
	decider: { kind: "jev", baseUrl: "", model: "", chat: "anthropic/claude-opus-5-5" },
	planner: { provider: "anthropic", model: "claude-opus-5-5" },
	uiPort: 8760,
	runsDir: "runs",
};

type Table = Record<string, any>;

export function loadConfig(path?: string): Config {
	const file = path ?? (existsSync("valet.toml") ? "valet.toml" : undefined);
	const t: Table = file ? parse(readFileSync(file, "utf8")) : {};
	const d = DEFAULTS;
	const kind = t.decider?.kind ?? d.decider.kind;
	if (!DECIDER_KINDS.includes(kind)) throw new Error(`[decider] kind must be one of ${DECIDER_KINDS.join(", ")}, got ${JSON.stringify(kind)}`);
	return {
		armdUrl: t.armd?.url ?? d.armdUrl,
		viewpoints: { wrist: t.cameras?.wrist_view ?? d.viewpoints.wrist, side: t.cameras?.side_view ?? d.viewpoints.side },
		decider: {
			kind,
			baseUrl: t.decider?.base_url ?? d.decider.baseUrl,
			model: t.decider?.model ?? d.decider.model,
			chat: t.decider?.chat ?? d.decider.chat,
		},
		planner: { provider: t.planner?.provider ?? d.planner.provider, model: t.planner?.model ?? d.planner.model },
		uiPort: t.ui?.port ?? d.uiPort,
		runsDir: d.runsDir,
	};
}

export function loadEnv(cwd = process.cwd()): string | undefined {
	const candidates = [join(cwd, ".env")];
	const main = mainCheckout(cwd);
	if (main) candidates.push(join(main, ".env"));
	const found = candidates.find((f) => existsSync(f));
	if (found) process.loadEnvFile(found);
	return found;
}

function mainCheckout(cwd: string): string | undefined {
	const dotGit = join(cwd, ".git");
	if (!existsSync(dotGit)) return undefined;
	const m = /^gitdir:\s*(.+)$/m.exec(safeRead(dotGit));
	if (!m) return undefined;
	const worktreesDir = dirname(m[1]!.trim());
	return dirname(dirname(worktreesDir));
}

function safeRead(file: string): string {
	try {
		return readFileSync(file, "utf8");
	} catch {
		return "";
	}
}
