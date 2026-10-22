// What the viewer says about a saved lock beyond /api/locks' summary: its
// train / held-out split, what kind of fit it is, whether the run was cut
// short, and a readable name. The two requests here call existing routes
// (GET and DELETE /api/locks/{name}); api.ts is left as the API contract.

import { ApiError, type LockSummary, type Params } from "./api";

// /api/locks also returns the fit's search space; api.ts's type predates it.
export type LockRow = LockSummary & { space?: string };

export type LockStage = { name: string; n_evaluations?: number; seed?: number; train_score?: number };

// The parts of GET /api/locks/{name} (the whole lock file) the viewer reads.
export type LockDetail = {
  name: string;
  command_map?: unknown;
  result?: {
    eval_episodes?: number[];
    train_episodes?: number[];
    n_evaluations?: number;
    space?: string;
    stages?: LockStage[];
  };
};

async function ok<T>(r: Response): Promise<T> {
  if (r.ok) return r.json() as Promise<T>;
  let body: unknown = null;
  try {
    body = await r.json();
  } catch {
    // not JSON
  }
  const detail = (body as { detail?: unknown } | null)?.detail;
  throw new ApiError(r.status, detail ?? `${r.status} ${r.statusText}`);
}

export const lockApi = {
  get: (name: string) => fetch(`/api/locks/${encodeURIComponent(name)}`).then((r) => ok<LockDetail>(r)),
  // Unlinks the lock file at once; the API has no restore, so the viewer
  // only calls this after the undo window (see App's soft delete).
  remove: (name: string) => fetch(`/api/locks/${encodeURIComponent(name)}`, { method: "DELETE" }).then((r) => ok<{ deleted: string }>(r)),
};

// ---- names ------------------------------------------------------------------------

// antares.lock.slug(dataset), as lockNameFor in api.ts builds it.
export function datasetSlug(dataset: string): string {
  const name = dataset.replace(/\/+$/, "").split("/").pop()!.trim().toLowerCase();
  return name.replace(/[^a-z0-9_.-]+/g, "_").replace(/^[_.-]+|[_.-]+$/g, "") || "robot";
}

// The timestamp lockNameFor appends to a lock calibrated from the viewer.
const STAMP = /-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/;

export const isViewerRun = (name: string) => STAMP.test(name);

const sentence = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);

// "svla_so101_pickplace-20261018-015353" -> "Run 01:53:53";
// "svla_so101_pickplace_perjoint" -> "Per-joint"; the dataset's own lock keeps its name.
export function lockLabel(name: string, dataset: string): string {
  const m = name.match(STAMP);
  if (m) return `Run ${m[4]}:${m[5]}:${m[6]}`;
  const slug = datasetSlug(dataset);
  if (name.length > slug.length + 1 && name.startsWith(slug) && /[_-]/.test(name[slug.length])) {
    const rest = name
      .slice(slug.length + 1)
      .replace(/[_-]+/g, " ")
      .replace(/\bperjoint\b/i, "per-joint")
      .trim();
    if (rest) return sentence(rest);
  }
  return name;
}

// "today 01:53", "yesterday 22:54", "Oct 12 09:30" in local time. With
// `dateOnly` (a viewer run, whose name already has the time): "", "yesterday", "Oct 12".
export function fmtWhen(iso: string | null | undefined, dateOnly = false): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const hm = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false });
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(new Date()) - day(d)) / 86400000);
  const date = diff === 0 ? (dateOnly ? "" : "today") : diff === 1 ? "yesterday" : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return dateOnly ? date : `${date} ${hm}`;
}

// ---- kind -------------------------------------------------------------------------------

export type LockKind = "shared" | "per-joint" | "sensor";

// A sensor fit also maps the encoder readings (command_map); a per-joint fit
// has one value per joint. Both can apply; "shared" only when neither does.
export function lockKinds(row: LockRow, detail?: LockDetail): LockKind[] {
  const kinds: LockKind[] = [];
  if (detail?.command_map || detail?.result?.stages?.some((s) => s.name === "command_map")) kinds.push("sensor");
  if ((row.space ?? detail?.result?.space) === "per_joint") kinds.push("per-joint");
  return kinds.length ? kinds : ["shared"];
}

export const KIND_HELP: Record<LockKind, string> = {
  shared: "One set of values for all joints.",
  "per-joint": "One value per joint for each parameter.",
  sensor: "Also maps the encoder readings (a sensor calibration).",
};

// ---- partial runs ------------------------------------------------------------------------

// Budgets of a calibration started from the viewer (App's MAX_EVALUATIONS and
// PER_JOINT_EVALUATIONS); the API does not store what a run asked for.
export const VIEWER_SHARED_BUDGET = 150;
export const VIEWER_PER_JOINT_BUDGET = 600;

export type PartialRun = { n: number; total: number };

const PARTIAL_KEY = "antares.partial-runs";

// Runs stopped from this browser, with the evaluations they had done and asked for.
export function readPartialRuns(): Record<string, PartialRun> {
  try {
    const v = JSON.parse(window.localStorage.getItem(PARTIAL_KEY) ?? "{}");
    return v && typeof v === "object" ? (v as Record<string, PartialRun>) : {};
  } catch {
    return {};
  }
}

export function recordPartialRun(name: string, p: PartialRun): Record<string, PartialRun> {
  const all = { ...readPartialRuns(), [name]: p };
  try {
    window.localStorage.setItem(PARTIAL_KEY, JSON.stringify(all));
  } catch {
    // storage unavailable: the badge still shows from the heuristic below
  }
  return all;
}

// A lock is partial when its run was stopped before its budget: known
// exactly for runs stopped from this browser, and for any viewer run
// (timestamped name) whose evaluation count is below the viewer's budget.
export function partialOf(row: LockRow, detail: LockDetail | undefined, stopped: Record<string, PartialRun>): PartialRun | null {
  const rec = stopped[row.name];
  if (rec && rec.n < rec.total) return rec;
  if (!isViewerRun(row.name)) return null;
  const n = row.n_evaluations ?? detail?.result?.n_evaluations;
  if (n == null) return null;
  const perJoint = detail?.result?.stages?.some((s) => s.name === "per_joint") || row.space === "per_joint";
  const total = VIEWER_SHARED_BUDGET + (perJoint ? VIEWER_PER_JOINT_BUDGET : 0);
  return n < total ? { n, total } : null;
}

// ---- repeats -------------------------------------------------------------------------------

function sameParams(a: Params, b: Params): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]) as Set<keyof Params>;
  for (const k of keys) {
    const x = a[k];
    const y = b[k];
    if (JSON.stringify(x) === JSON.stringify(y)) continue;
    if (typeof x === "number" && typeof y === "number" && Math.abs(x - y) <= 1e-12 * Math.max(1, Math.abs(x))) continue;
    return false;
  }
  return true;
}

// An older lock of the same dataset with the same parameters and held-out
// error: calibration is deterministic, so the same options give the same lock.
export function repeatOf(row: LockRow, all: LockRow[]): LockRow | null {
  const t = Date.parse(row.created_at);
  // The earliest such lock, so every repeat points at the same original.
  const oldestFirst = [...all].sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at));
  return (
    oldestFirst.find(
      (o) =>
        o.name !== row.name &&
        o.dataset === row.dataset &&
        Date.parse(o.created_at) < t &&
        o.scores.eval_score != null &&
        row.scores.eval_score != null &&
        Math.abs(o.scores.eval_score - row.scores.eval_score) < 1e-9 &&
        sameParams(o.params, row.params),
    ) ?? null
  );
}

// ---- splits and an episode's role --------------------------------------------------------

export type Split = { train: number[]; eval: number[] };
export type EpisodeRole = "held out" | "train" | "unused";

export function splitOf(detail: LockDetail | undefined): Split | null {
  const r = detail?.result;
  if (!r?.eval_episodes || !r.train_episodes) return null;
  return { train: r.train_episodes, eval: r.eval_episodes };
}

// The API's default split (20 train / 10 held out, seed 0) is what every
// viewer calibration uses, and the residual baseline of the stock model too.
// The client cannot recompute numpy's permutation, so it reads the split a
// viewer run recorded, else the split most of the dataset's locks share.
export function defaultSplit(rows: LockRow[], details: Record<string, LockDetail | undefined>): Split | null {
  const viewer = rows.find((r) => isViewerRun(r.name) && splitOf(details[r.name]));
  if (viewer) return splitOf(details[viewer.name]);
  const counts = new Map<string, { split: Split; n: number }>();
  for (const r of rows) {
    const s = splitOf(details[r.name]);
    if (!s || s.train.length !== 20 || s.eval.length !== 10) continue;
    const key = JSON.stringify(s);
    counts.set(key, { split: s, n: (counts.get(key)?.n ?? 0) + 1 });
  }
  let best: { split: Split; n: number } | null = null;
  for (const c of counts.values()) if (!best || c.n > best.n) best = c;
  return best?.split ?? null;
}

export function roleOf(split: Split | null, episode: number | null | undefined): EpisodeRole | null {
  if (!split || episode == null) return null;
  if (split.eval.includes(episode)) return "held out";
  if (split.train.includes(episode)) return "train";
  return "unused";
}

export const ROLE_HELP: Record<EpisodeRole, string> = {
  "held out": "Held out: not used in the fit, so its error is an honest test.",
  train: "Training episode: the fit saw it, so its error is optimistic.",
  unused: "Not in the split: neither trained on nor among the held-out episodes.",
};
