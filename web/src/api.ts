// Thin client for the antares API (antares/server.py). Angles are degrees,
// poses are MuJoCo world-frame [x, y, z, qw, qx, qy, qz] per body.

// A parameter is one shared value or, for a per-joint lock, one value per joint.
export type ParamValue = number | number[];

export type Params = {
  kp: ParamValue;
  kv: ParamValue;
  damping: ParamValue;
  armature: ParamValue;
  frictionloss: ParamValue;
  latency_frames: ParamValue;
  // Actuator force limit; searched by the per-joint stage only.
  forcerange?: ParamValue;
};

export type DatasetInfo = {
  id: string;
  loaded: boolean;
  robot_type?: string;
  fps?: number;
  n_episodes?: number;
  joints?: string[];
};

export type EpisodeInfo = { index: number; frames: number; duration_s: number };

export type Geometry = {
  bodies: { id: number; name: string; parent: number }[];
  geoms: { body: number; mesh: string; pos: number[]; quat: number[]; rgba: number[] }[];
  joints: { name: string; body: number; axis: number[]; range: number[] }[];
};

export type Gap = {
  per_joint_deg: Record<string, number>;
  overall_deg: number;
  scored_joints: string[];
  n_frames: number;
};

export type Replay = {
  dataset: string;
  episode: number;
  fps: number;
  joints: string[];
  body_names: string[];
  params: Params;
  real_q: number[][];
  sim_q: number[][];
  real_poses: number[][][];
  sim_poses: number[][][];
  gap: Gap;
  lock: string | null;
};

export type LockSummary = {
  name: string;
  dataset: string;
  robot_type: string;
  created_at: string;
  params: Params;
  scores: {
    train_score: number | null;
    eval_score: number | null;
    baseline_train_score: number | null;
    baseline_eval_score: number | null;
  };
  n_evaluations: number | null;
  elapsed_s: number | null;
};

export type ProgressEvent = {
  type: "progress";
  // Which CMA-ES search space produced this evaluation; `evaluation` keeps
  // counting across stages.
  stage: "shared" | "per_joint" | "command_map";
  evaluation: number;
  generation: number;
  candidate_score: number | null;
  best_score: number | null;
  best_params: Params;
  best_per_joint_deg: Record<string, number | null>;
  elapsed_s: number;
};

export type CalibrationResult = {
  params: Params;
  train_score: number;
  eval_score: number;
  baseline_train_score: number;
  baseline_eval_score: number;
  n_evaluations: number;
  elapsed_s: number;
};

export type ResidualEvent = {
  start: number; // half-open [start, end) frame range
  end: number;
  joints: string[];
  peak_z: number;
  peak_deg: number;
  kind: "contact" | "anomaly";
};

// GET /api/residual: real - sim per frame for one episode, with the events
// where the sim disagrees with the arm and the gripper's grasp windows.
export type Residual = {
  dataset: string;
  episode: number;
  fps: number;
  lock: string | null;
  joints: string[];
  scored_joints: string[];
  z_threshold: number;
  warmup_frames: number;
  residual: number[][]; // (T, 6) deg
  z: number[][]; // (T, 6)
  events: ResidualEvent[];
  gripper_cmd: number[];
  grasps: number[];
  releases: number[];
  grasp_windows: number[][]; // [[start, end], ...]
};

// GET /api/compensate: commands that make the calibrated sim follow the real
// trajectory, judged in the sim only (no hardware was driven).
export type Compensation = {
  dataset: string;
  episode: number;
  fps: number;
  lock: string | null;
  method: "shift" | "ilc";
  validation: "sim-only";
  note: string;
  sim_comp_poses: number[][][]; // (T, nbody, 7)
  rms: { raw: number; shift: number; ilc: number }; // deg, gripper excluded
  ilc_iterations: number;
};

// GET /api/cross: every lock (plus the default sim) scored on every dataset's
// held-out split, RMS degrees.
export type CrossMatrix = {
  rows: string[]; // datasets
  cols: string[]; // locks + "default"
  scores: number[][]; // [row][col]
  per_joint: Record<string, number>[][];
};

export type JobEvent =
  | ProgressEvent
  | { type: "done"; lock: { name: string; params: Params }; result: CalibrationResult }
  | { type: "error"; message: string };

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // not JSON
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

const q = (o: Record<string, string | number>) =>
  new URLSearchParams(Object.entries(o).map(([k, v]) => [k, String(v)])).toString();

export const api = {
  datasets: () => fetch("/api/datasets").then((r) => json<DatasetInfo[]>(r)),
  loadDataset: (dataset: string) =>
    fetch(`/api/datasets/load?${q({ dataset })}`, { method: "POST" }).then((r) => json<DatasetInfo>(r)),
  episodes: (dataset: string) => fetch(`/api/episodes?${q({ dataset })}`).then((r) => json<EpisodeInfo[]>(r)),
  model: () => fetch("/api/model").then((r) => json<Geometry>(r)),
  meshUrl: (file: string) => `/api/model/mesh/${encodeURIComponent(file)}`,
  locks: () => fetch("/api/locks").then((r) => json<LockSummary[]>(r)),
  replay: (dataset: string, episode: number, lock: string | null) =>
    fetch(`/api/replay?${q(lock ? { dataset, episode, lock } : { dataset, episode })}`).then((r) =>
      json<Replay>(r),
    ),
  replayParams: (dataset: string, episode: number, params: Params) =>
    fetch("/api/replay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, episode, params }),
    }).then((r) => json<Replay>(r)),
  // `perJoint` adds the per-joint refinement stage after the shared fit.
  calibrate: (dataset: string, name: string, maxEvaluations: number, perJoint?: { maxEvaluations: number }) =>
    fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset,
        name,
        max_evaluations: maxEvaluations,
        ...(perJoint ? { per_joint: true, per_joint_max_evaluations: perJoint.maxEvaluations } : {}),
      }),
    }).then((r) => json<{ job_id: string }>(r)),
  cancel: (jobId: string) => fetch(`/api/calibrate/${jobId}/cancel`, { method: "POST" }).then((r) => json(r)),
  events: (jobId: string) => new EventSource(`/api/calibrate/${jobId}/events`),
  residual: (dataset: string, episode: number, lock: string | null) =>
    fetch(`/api/residual?${q(lock ? { dataset, episode, lock } : { dataset, episode })}`).then((r) =>
      json<Residual>(r),
    ),
  compensate: (dataset: string, episode: number, lock: string | null, method: "shift" | "ilc" = "ilc") =>
    fetch(`/api/compensate?${q(lock ? { dataset, episode, lock, method } : { dataset, episode, method })}`).then(
      (r) => json<Compensation>(r),
    ),
  cross: (datasets: string[], locks: string[], nTrain = 20, nEval = 10) =>
    fetch(
      `/api/cross?${q({ datasets: datasets.join(","), locks: locks.join(","), n_train: nTrain, n_eval: nEval })}`,
    ).then((r) => json<CrossMatrix>(r)),
  // Zip of the lock as a standalone MJCF (model, sidecar, scene, meshes).
  exportZipUrl: (lock: string) => `/api/locks/${encodeURIComponent(lock)}/export.zip`,
};

// antares.lock.slug(dataset) plus a timestamp, so a run from the UI never
// overwrites the committed demo lock.
export function lockNameFor(dataset: string, when = new Date()): string {
  const name = dataset.replace(/\/+$/, "").split("/").pop()!.trim().toLowerCase();
  const base = name.replace(/[^a-z0-9_.-]+/g, "_").replace(/^[_.-]+|[_.-]+$/g, "") || "robot";
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = `${when.getFullYear()}${p(when.getMonth() + 1)}${p(when.getDate())}-${p(when.getHours())}${p(when.getMinutes())}${p(when.getSeconds())}`;
  return `${base}-${stamp}`;
}
