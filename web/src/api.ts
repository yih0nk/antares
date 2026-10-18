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

// ---- cameras (antares.camera): a dataset's fixed cameras fitted to the logs ----

// GET /api/camera/cameras: the dataset's video streams.
export type CameraInfo = {
  name: string; // "up"
  key: string; // "observation.images.up"
  width: number;
  height: number;
  fps: number;
  fixed: boolean; // false for wrist cameras, which cannot be fitted
};

// One fitted camera as saved in locks/<lock>.camera.json (per-frame arrays omitted).
export type CameraFit = {
  camera: string;
  key: string;
  dataset: string;
  pos: number[];
  rotvec: number[];
  quat: number[]; // MuJoCo wxyz
  fovy: number;
  width: number;
  height: number;
  fit_iou: number;
  heldout_iou: number;
  fit_frames: number[][]; // [episode, frame]
  heldout_frames: number[][];
  n_starts: number;
  best_start: number;
  start_ious: number[];
  n_evaluations: number;
  elapsed_s: number;
  cancelled: boolean;
  created_at: string;
};

export type CameraProgressEvent = {
  type: "progress";
  stage: "sample" | "start" | "refine" | "heldout";
  start: number; // index of the start being run (-1 while sampling)
  n_starts: number;
  evaluation: number;
  stage_evaluation: number;
  candidate_iou: number;
  stage_best_iou: number;
  best_iou: number;
  elapsed_s: number;
};

export type CameraJobEvent =
  | CameraProgressEvent
  | { type: "done"; fit: CameraFit & { lock: string; path: string } }
  | { type: "error"; message: string };

// GET /api/camera/overlay: one real frame with the sim arm's outlines in the
// fitted camera. Polygons are in pixel coordinates of the frame.
export type CameraOverlay = {
  dataset: string;
  camera: string;
  episode: number;
  frame: number;
  lock: string | null;
  width: number;
  height: number;
  image: string; // data: URL (JPEG)
  sim_polygons: number[][][]; // per visible geom, [[x, y], ...]
  sim_bodies?: string[];
  real_mask_polygon?: number[][][];
  fit: { fit_iou: number; heldout_iou: number; fovy: number; pos: number[]; quat: number[]; created_at: string } | null;
  iou: number | null;
};

// ---- scene (antares.scene): table, workspace and backdrops from the logs ----

export type SceneTable = {
  z: number; // m, at the arm's base
  normal: number[];
  tilt_deg: number;
  horizontal: boolean;
  n_grasps: number;
  n_on_table: number;
  z_grasp: number;
  object_half_height: number;
  fingertip_z: Record<string, number>;
  jaw_low_z: Record<string, number>;
  episode_min_z: Record<string, number>;
  min_fingertip_z: number;
};

export type SceneWorkspace = { min: number[]; max: number[]; size: number[]; hull_xy: number[][]; n_frames: number };

// One camera's backdrop and how the projected table plane sits in its image.
export type SceneBackdrop = {
  camera: string;
  path: string;
  width: number;
  height: number;
  plane_polygon: number[][]; // pixels
  plane_frac: number;
  region_frac: number;
  plane_in_region: number; // share of the projected plane on the dominant colour region
  region_in_plane: number;
  iou: number;
  table_rgb: number[];
};

// GET /api/scene?lock=: locks/<lock>.scene.json.
export type SceneFit = {
  lock: string;
  dataset: string;
  robot: string;
  table: SceneTable;
  workspace: SceneWorkspace;
  grasp_points: number[][];
  release_points: number[][];
  backdrops: Record<string, SceneBackdrop>;
  n_episodes: number;
  n_frames: number;
  created_at: string;
};

// GET /api/scene/view: the real frame and the sim's view of the same instant
// from the same camera (backdrop + table + flat-shaded arm).
export type SceneView = {
  dataset: string;
  camera: string;
  episode: number;
  frame: number;
  lock: string;
  width: number;
  height: number;
  real_image: string; // data: URL (JPEG)
  sim_image: string;
  iou: number;
  table_polygon: number[][];
  table_z: number;
  n_geoms: number;
};

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
  // Zip of the lock as a standalone MJCF (model, sidecar, scene, meshes).
  exportZipUrl: (lock: string) => `/api/locks/${encodeURIComponent(lock)}/export.zip`,
  // ---- cameras ----
  cameras: (dataset: string) => fetch(`/api/camera/cameras?${q({ dataset })}`).then((r) => json<CameraInfo[]>(r)),
  cameraFits: (lock: string) =>
    fetch(`/api/camera?${q({ lock })}`).then((r) => json<{ lock: string; cameras: Record<string, CameraFit> }>(r)),
  cameraFit: (dataset: string, camera: string, lock: string, nFrames = 40, nEpisodes = 4) =>
    fetch("/api/camera/fit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, camera, lock, n_frames: nFrames, n_episodes: nEpisodes }),
    }).then((r) => json<{ job_id: string }>(r)),
  cameraOverlay: (dataset: string, camera: string, episode: number, frame: number, lock: string | null) =>
    fetch(
      `/api/camera/overlay?${q(lock ? { dataset, camera, episode, frame, lock } : { dataset, camera, episode, frame })}`,
    ).then((r) => json<CameraOverlay>(r)),
  // ---- scene ----
  scene: (lock: string) =>
    fetch(`/api/scene?${q({ lock })}`).then((r) => json<{ lock: string; scene: SceneFit | null }>(r)),
  sceneFit: (dataset: string, lock: string) =>
    fetch("/api/scene/fit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, lock }),
    }).then((r) => json<SceneFit & { path: string }>(r)),
  sceneView: (dataset: string, camera: string, episode: number, frame: number, lock: string) =>
    fetch(`/api/scene/view?${q({ dataset, camera, episode, frame, lock })}`).then((r) => json<SceneView>(r)),
  jobEvents: (jobId: string) => new EventSource(`/api/jobs/${jobId}/events`),
  cancelJob: (jobId: string) => fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" }).then((r) => json(r)),
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
