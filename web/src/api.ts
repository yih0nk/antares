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

// Where a dataset came from: pinned demo, any Hub id, a zip/parquet upload,
// or a directory on the server's disk.
export type DatasetSource = "demo" | "hub" | "upload" | "local";

export type DatasetInfo = {
  id: string;
  loaded: boolean;
  source: DatasetSource;
  // Validation warnings (few episodes, odd units, a name that smells like a sim).
  warnings: string[];
  // What the dataset was recorded on, and which registry model simulates it.
  robot_family: "so100" | "so101" | null;
  robot: string;
  robot_type?: string;
  fps?: number;
  n_episodes?: number;
  joints?: string[];
};

// GET /api/datasets/validate: can the sim replay this dataset? Decided from
// meta/info.json alone, before anything is downloaded.
export type Verdict = {
  ok: boolean;
  robot_type: string;
  robot_family: "so100" | "so101" | null;
  reasons: string[];
  warnings: string[];
  fps: number;
  n_episodes: number;
  n_frames: number;
  joint_names: string[];
  codebase_version: string;
};

export type ValidateResponse = Verdict & { dataset: string; loaded: boolean; source: DatasetSource; robot: string };

// POST /api/datasets/upload: the dataset summary plus the verdict it passed.
export type UploadResponse = DatasetInfo & { verdict: Verdict };

export type EpisodeInfo = { index: number; frames: number; duration_s: number };

export type Geometry = {
  robot: string;
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
  // Registry name of the MuJoCo model that produced the ghost.
  robot: string;
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

// One `antares check` run as the history store records it; `verdict` is the
// gate's outcome and `score` the fresh episodes' RMS error against `limit`.
export type CheckVerdict = "ok" | "drifted" | "insufficient_data";

export type HistoryEntry = {
  at: string;
  lock: string;
  dataset: string;
  verdict: CheckVerdict;
  passed: boolean;
  score: number | null;
  reference: number | null;
  limit: number | null;
  n_episodes: number;
  episodes: number[];
  worst_joint: string | null;
  per_joint: Record<string, number | null>;
  antares_version: string;
};

// Least-squares line through a lock's check scores over time.
export type HistoryTrend = {
  n: number;
  first: string | null;
  last: string | null;
  slope_deg_per_day: number | null;
  r2: number | null;
  latest_verdict: CheckVerdict | string | null;
  n_drifted: number;
};

// GET /api/history/{lock}: every check of one lock, oldest first.
export type LockHistory = {
  lock: string;
  entries: HistoryEntry[];
  trend: HistoryTrend | null;
};

// GET /api/dashboard: one row per saved lock, with what the fleet view needs
// joined in (drift scan, fleet membership, latest check and trend).
export type DashboardRobot = {
  name: string;
  dataset: string | null;
  robot_type: string | null;
  robot: string;
  created_at: string | null;
  space: "shared" | "per_joint" | string;
  baseline_eval_score: number | null;
  eval_score: number | null;
  improvement: number | null;
  per_joint_after: Record<string, number | null> | null;
  worst_joint: string | null;
  worst_joint_deg: number | null;
  latency_frames: ParamValue | null;
  // The dataset's frame rate when the server knows it (fleet report or a
  // loaded dataset); turns latency frames into milliseconds.
  fps: number | null;
  drift: { exists: boolean; verdict: string | null };
  in_fleet: boolean;
  // Parameters a seeded calibration held fixed, when the lock records them.
  pinned: string[] | null;
  history: { latest: HistoryEntry | null; trend: HistoryTrend | null };
};

export type FleetSummary = {
  n_robots: number;
  median_improvement?: number | null;
  created_at?: string | null;
  locks?: string[];
  [k: string]: unknown;
};

export type Dashboard = {
  robots: DashboardRobot[];
  fleet: FleetSummary | null;
  generated_at: string;
};

export type JobEvent =
  | ProgressEvent
  | { type: "done"; lock: { name: string; params: Params }; result: CalibrationResult }
  | { type: "error"; message: string };

// A failed request. `detail` is the API's error body: a string, or for a
// rejected dataset `{ message, verdict }` so the UI can list the reasons.
export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : ((detail as { message?: string })?.message ?? `${status}`));
    this.status = status;
    this.detail = detail;
  }
  get verdict(): Verdict | null {
    const d = this.detail as { verdict?: Verdict } | null;
    return d && typeof d === "object" && d.verdict ? d.verdict : null;
  }
}

function parseDetail(status: number, statusText: string, body: unknown): ApiError {
  const detail = (body as { detail?: unknown } | null)?.detail;
  return new ApiError(status, detail ?? `${status} ${statusText}`);
}

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let body: unknown = null;
    try {
      body = await r.json();
    } catch {
      // not JSON
    }
    throw parseDetail(r.status, r.statusText, body);
  }
  return r.json() as Promise<T>;
}

const q = (o: Record<string, string | number>) =>
  new URLSearchParams(Object.entries(o).map(([k, v]) => [k, String(v)])).toString();

// Multipart upload with progress; fetch() cannot report upload progress.
function uploadWithProgress<T>(url: string, form: FormData, onProgress: (fraction: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) onProgress(ev.loaded / ev.total);
    };
    xhr.onerror = () => reject(new ApiError(0, "network error during upload"));
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // not JSON
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else reject(parseDetail(xhr.status, xhr.statusText, body));
    };
    xhr.send(form);
  });
}

export const api = {
  datasets: () => fetch("/api/datasets").then((r) => json<DatasetInfo[]>(r)),
  loadDataset: (dataset: string) =>
    fetch(`/api/datasets/load?${q({ dataset })}`, { method: "POST" }).then((r) => json<DatasetInfo>(r)),
  // Cheap pre-check of a Hub id (meta/info.json only).
  validateDataset: (dataset: string) =>
    fetch(`/api/datasets/validate?${q({ dataset })}`).then((r) => json<ValidateResponse>(r)),
  // A .zip of a LeRobot dataset directory; `name` defaults to the file name.
  uploadDataset: (file: File, onProgress: (fraction: number) => void, name?: string) => {
    const form = new FormData();
    form.append("file", file, file.name);
    if (name) form.append("name", name);
    return uploadWithProgress<UploadResponse>("/api/datasets/upload", form, onProgress);
  },
  deleteDataset: (dataset: string) =>
    fetch(`/api/datasets?${q({ dataset })}`, { method: "DELETE" }).then((r) => json<{ deleted: string }>(r)),
  episodes: (dataset: string) => fetch(`/api/episodes?${q({ dataset })}`).then((r) => json<EpisodeInfo[]>(r)),
  // Geometry of one registry model (default so101); meshes are fetched per robot too.
  model: (robot?: string) => fetch(robot ? `/api/model?${q({ robot })}` : "/api/model").then((r) => json<Geometry>(r)),
  meshUrl: (file: string, robot?: string) =>
    `/api/model/mesh/${encodeURIComponent(file)}${robot ? `?${q({ robot })}` : ""}`,
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
  dashboard: () => fetch("/api/dashboard").then((r) => json<Dashboard>(r)),
  // A lock that was never checked (or a server without the history store)
  // answers 404; the fleet view shows that as an empty history.
  history: (lock: string) =>
    fetch(`/api/history/${encodeURIComponent(lock)}`).then((r) =>
      r.status === 404 ? ({ lock, entries: [], trend: null } satisfies LockHistory) : json<LockHistory>(r),
    ),
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
