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
  // Gravity load (fraction of the MJCF's link weights; 1 is the stock model)
  // and servo deadband (rad). Fitted with fit_gravity / fit_deadband and
  // omitted at their defaults, so older locks have neither.
  gravity_scale?: ParamValue;
  deadband?: ParamValue;
  // Actuator model: absent or "position" is MuJoCo's position actuator;
  // "sts3215" simulates the Feetech servo's firmware and motor. With a
  // servo model the fields below are written too, and kp/kv/forcerange hold
  // the model's linearized PD equivalent rather than fitted values.
  servo?: string;
  servo_gain?: ParamValue;
  p_coefficient?: ParamValue;
  vin?: ParamValue;
  kt?: ParamValue;
  resistance?: ParamValue;
  max_velocity?: ParamValue;
};

// The numeric parameters (everything but the actuator model's name).
export type NumericParamKey = Exclude<keyof Params, "servo">;

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

export type FactorInfo = {
  name: string;
  label: string;
  explains: string;
  unit: string;
  bounds: [number, number];
  needs: string;
  identifiable: "yes" | "partly" | "no" | "untested";
  evidence: string;
  stages: string[];
  origin: string;
};

export type FactorRegistry = {
  version: number;
  factors: FactorInfo[];
  not_modelled: { name: string; label: string; symptom: string; would_need: string }[];
};

export type Geometry = {
  robot: string;
  bodies: { id: number; name: string; parent: number }[];
  geoms: { body: number; mesh: string; pos: number[]; quat: number[]; rgba: number[]; scale?: number[] }[];
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
  // Registry (locks/registry.json): the physical robot this lock belongs to
  // and its place in that robot's lineage; all null when unregistered.
  robot_id: string | null;
  is_active: boolean | null;
  version: number | null;
  n_versions: number | null;
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

// The camera to show once a dataset's cameras are known: the one already
// chosen if it is still there, else the first fixed one, else the first.
// Shared by the Camera panel and the video underlay.
export function pickCamera(cameras: CameraInfo[], current: string): string {
  if (cameras.some((c) => c.name === current)) return current;
  return (cameras.find((c) => c.fixed) ?? cameras[0])?.name ?? "";
}

// The clip format this browser plays, checked once: H.264 mp4 everywhere but
// a Chromium build without H.264, which gets VP9 webm.
const CLIP_FORMAT: "mp4" | "webm" =
  typeof document !== "undefined" && document.createElement("video").canPlayType('video/mp4; codecs="avc1.42E01E"')
    ? "mp4"
    : "webm";

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

// ---- policy (antares.policy / shadow / closedloop): a learned policy in the loop ----

export type PolicyMode = "shadow" | "closedloop";

// Chunk-prediction errors of a predictor on one split (antares.policy.action_errors).
export type PolicyErrors = {
  mse_units: number;
  rmse_units: number;
  rmse_deg: number; // RMS over the body joints, whole chunk
  first_rmse_deg: number; // first action of the chunk only
  gripper_rmse_units: number;
  per_joint_deg: Record<string, number>;
  n_windows: number;
};

export type PolicyMetrics = {
  policy: { train: PolicyErrors; eval: PolicyErrors };
  baselines: Record<"hold_state" | "repeat_command", { train: PolicyErrors; eval: PolicyErrors }>;
  best_epoch: number;
  epochs_run: number;
  beats_hold_state: boolean;
  beats_repeat_command: boolean;
};

// locks/<name>.policy.json: a saved policy and how it trained.
export type PolicyInfo = {
  name: string;
  lock?: string;
  dataset?: string;
  k: number;
  h: number;
  features: { goal: boolean; time: boolean };
  n_params: number;
  train_episodes: number[];
  eval_episodes: number[];
  metrics: PolicyMetrics;
  elapsed_s: number;
  created_at: string;
  exists: boolean;
};

export type PolicyTrainProgress = {
  type: "progress";
  epoch: number;
  epochs: number;
  train_mse: number; // normalized units
  eval_mse: number;
  best_eval_mse: number;
  best_epoch: number;
  elapsed_s: number;
};

export type PolicyTrainEvent = PolicyTrainProgress | { type: "done"; policy: PolicyInfo } | { type: "error"; message: string };

export type ShadowVariantSummary = {
  sim_vs_real_deg: number;
  replay_vs_real_deg: number;
  grasp_error_cm: number | null;
  n_grasps: number;
  // Only for the non-baseline sims: how far they are from the default sim's trajectory.
  divergence_deg?: number;
  grasp_divergence_cm?: number | null;
  worst_frames?: { episode: number; frame: number; deg: number }[];
};

export type ShadowResult = {
  mode: "shadow";
  baseline: string;
  lock: string;
  policy: string;
  variants: { name: string; label: string; lock: string | null }[];
  summary: {
    n_episodes: number;
    n_frames: number;
    action_error_deg: number;
    variants: Record<string, ShadowVariantSummary>;
  };
  episodes: { episode: number; n_frames: number; action_error_deg: number; sim_vs_real_deg: Record<string, number>; divergence_deg: Record<string, number> }[];
  episodes_evaluated: number[];
  created_at: string;
};

export type ClosedLoopVariantSummary = {
  trajectory_rms_deg: number;
  approach_rms_deg: number | null;
  grasp_error_cm: number | null;
  min_tip_distance_cm: number | null;
  close_frame_offset: number | null;
  n_judged: number;
  n_closed: number;
  n_success: number;
  success_rate: number | null;
};

export type ClosedLoopRun = {
  variant: string;
  episode: number;
  trajectory_rms_deg: number;
  approach_rms_deg: number | null;
  grasp_error_cm: number | null;
  success: boolean | null;
  close_frame: number | null;
  real_grasp_frame: number | null;
};

export type ClosedLoopResult = {
  mode: "closedloop";
  lock: string;
  policy: string;
  success_cm: number;
  run_names: string[];
  variants: { name: string; label: string; lock: string | null }[];
  summary: { n_episodes: number; variants: Record<string, ClosedLoopVariantSummary> };
  runs: ClosedLoopRun[];
  episodes_evaluated: number[];
  created_at: string;
};

// GET /api/policy/results?lock=: the saved evaluations of a lock's policy.
export type PolicyResults = { lock: string; shadow: ShadowResult | null; closedloop: ClosedLoopResult | null };

export type PolicyEvalEvent =
  | { type: "progress"; mode: PolicyMode; i: number; n: number; episode: number }
  | { type: "done"; mode: PolicyMode; lock: string; policy: string }
  | { type: "error"; message: string };

// GET /api/policy/viewer: one held-out episode under the policy in each sim,
// with body poses for the viewer's ghosts (like /api/replay).
export type PolicySim = {
  q: number[][];
  poses: number[][][];
  sim_vs_real_deg?: number;
  divergence_deg?: number;
  trajectory_rms_deg?: number;
  grasp_error_cm?: number | null;
  success?: boolean | null;
  close_frame?: number | null;
};

export type PolicyViewer = {
  dataset: string;
  lock: string;
  policy: string;
  mode: PolicyMode;
  episode: number;
  fps: number;
  body_names: string[];
  goal: number[] | null;
  real_q: number[][];
  real_poses: number[][][];
  sims: Record<string, PolicySim>;
  variants: { name: string; label: string }[];
};


// ---- objects (antares.objects / antares.payload): the manipulated object ----

type Quartiles = { n: number; median: number; q1: number; q3: number; iqr: number; max?: number; std?: number };

// One episode's summary as saved in locks/<lock>.objects.json.
export type ObjectTrackSummary = {
  episode: number;
  camera: string;
  n_frames: number;
  n_visible: number;
  picks: number[][]; // [[grasp, release], ...] held windows
  start_pose: number[] | null;
  end_pose: number[] | null;
  end_source: "vision" | "fk" | "none";
  footprint_m: number[] | null;
  fk_grasp: number[] | null;
  fk_release: number[] | null;
  vision_vs_fk_m: number | null;
  end_vs_fk_m: number | null;
  note: string;
};

export type ObjectCloud = { n: number; mean: number[]; std: number[]; min: number[]; max: number[]; points: number[][] };

// The residual-based mass estimate; `verdict` says whether it is a measurement.
export type PayloadEstimate = {
  kp: Record<string, number>;
  mass_kg: number | null;
  mass_iqr_kg: number | null;
  mass: Quartiles | null;
  mass_by_joint: Record<string, Quartiles | null>;
  payload_deg: Record<string, Quartiles | null>;
  motion_deg: Record<string, Quartiles | null>;
  mass_per_step_kg: Record<string, number | null>;
  mass_noise_floor_kg: Record<string, number | null>;
  n_ok: number;
  n_episodes: number;
  verdict: string;
  assumptions: string[];
};

// GET /api/objects?lock=: locks/<lock>.objects.json.
export type ObjectsFit = {
  lock: string;
  dataset: string;
  camera: string;
  robot: string;
  table_z: number;
  object_half_height: number;
  prior: { lab: number[]; spread: number[]; area_px: number; rgb: number[]; n_samples: number };
  description: string;
  tracks: ObjectTrackSummary[];
  start: ObjectCloud | null;
  end: ObjectCloud | null;
  footprint_m: { long: Quartiles | null; short: Quartiles | null } | null;
  vision_vs_fk_m: Quartiles | null;
  end_vs_fk_m: Quartiles | null;
  vision_vs_fk_offset_m: number[] | null;
  vision_vs_fk_centred_m: Quartiles | null;
  cross_camera: Record<string, unknown> | null;
  payload: PayloadEstimate | null;
  n_episodes: number;
  n_tracked: number;
  elapsed_s: number;
  created_at: string;
};

// GET /api/objects/track: one episode frame by frame. `source` says where
// each position comes from; `centroid`/`bbox` are pixels in the fit's camera.
export type ObjectTrack = ObjectTrackSummary & {
  dataset: string;
  lock: string;
  rgb: number[];
  half_height: number;
  frames: number[];
  xyz: number[][]; // (T, 3) base frame, m
  visible: boolean[];
  source: ("vision" | "held" | "hidden")[];
  centroid: (number[] | null)[];
  bbox: (number[] | null)[]; // [x, y, w, h]
};

export type ObjectsProgressEvent =
  | { type: "progress"; stage: "prior"; i: number; n: number; elapsed_s: number; prior: ObjectsFit["prior"] }
  | { type: "progress"; stage: "track"; episode: number; i: number; n: number; elapsed_s: number; track: ObjectTrackSummary }
  | { type: "progress"; stage: "cross"; cross_camera: Record<string, unknown> }
  | { type: "progress"; stage: "payload"; episode: number; i: number; n: number; mass_kg: number | null };

export type ObjectsJobEvent =
  | ObjectsProgressEvent
  | { type: "done"; fit: ObjectsFit & { path: string } }
  | { type: "error"; message: string };

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
    `/api/model/mesh/${file.split("/").map(encodeURIComponent).join("/")}${robot ? `?${q({ robot })}` : ""}`,
  locks: () => fetch("/api/locks").then((r) => json<LockSummary[]>(r)),
  replay: (dataset: string, episode: number, lock: string | null) =>
    fetch(`/api/replay?${q(lock ? { dataset, episode, lock } : { dataset, episode })}`).then((r) =>
      json<Replay>(r),
    ),
  // `lock` keeps that saved lock's command map and calibration unit map, so
  // parameters tuned from a lock replay with the lock's own physics.
  replayParams: (dataset: string, episode: number, params: Params, lock: string | null = null) =>
    fetch("/api/replay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lock ? { dataset, episode, params, lock } : { dataset, episode, params }),
    }).then((r) => json<Replay>(r)),
  // `perJoint` adds the per-joint refinement stage after the shared fit.
  // `factors` (null: the defaults) picks what the shared fit accounts for;
  // `terms` adds the optional gravity load and servo deadband parameters.
  calibrate: (
    dataset: string,
    name: string,
    maxEvaluations: number,
    perJoint?: { maxEvaluations: number },
    factors?: string[] | null,
    terms?: { fitGravity?: boolean; fitDeadband?: boolean },
  ) =>
    fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset,
        name,
        max_evaluations: maxEvaluations,
        ...(perJoint ? { per_joint: true, per_joint_max_evaluations: perJoint.maxEvaluations } : {}),
        ...(factors ? { factors } : {}),
        ...(terms?.fitGravity ? { fit_gravity: true } : {}),
        ...(terms?.fitDeadband ? { fit_deadband: true } : {}),
      }),
    }).then((r) => json<{ job_id: string }>(r)),
  factors: () => fetch("/api/factors").then((r) => json<FactorRegistry>(r)),
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
  // ---- cameras ----
  cameras: (dataset: string) => fetch(`/api/camera/cameras?${q({ dataset })}`).then((r) => json<CameraInfo[]>(r)),
  cameraFits: (lock: string) =>
    fetch(`/api/camera?${q({ lock })}`).then((r) => json<{ lock: string; cameras: Record<string, CameraFit> }>(r)),
  // The best saved fit of each camera over all of the dataset's locks, with the lock it is saved under.
  cameraFitsForDataset: (dataset: string) =>
    fetch(`/api/camera/dataset?${q({ dataset })}`).then((r) =>
      json<{ dataset: string; cameras: Record<string, CameraFit & { lock: string }> }>(r),
    ),
  cameraFit: (dataset: string, camera: string, lock: string, nFrames = 40, nEpisodes = 4) =>
    fetch("/api/camera/fit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, camera, lock, n_frames: nFrames, n_episodes: nEpisodes }),
    }).then((r) => json<{ job_id: string }>(r)),
  // One episode of a camera as a clip whose frame 0 is the episode's frame 0:
  // H.264 mp4, or VP9 webm for a browser without H.264.
  cameraClipUrl: (dataset: string, camera: string, episode: number) =>
    `/api/camera/clip?${q({ dataset, camera, episode, format: CLIP_FORMAT })}`,
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
  // ---- objects ----
  objects: (lock: string) =>
    fetch(`/api/objects?${q({ lock })}`).then((r) => json<{ lock: string; objects: ObjectsFit | null }>(r)),
  objectsFit: (dataset: string, lock: string, camera?: string) =>
    fetch("/api/objects/fit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(camera ? { dataset, lock, camera } : { dataset, lock }),
    }).then((r) => json<{ job_id: string }>(r)),
  objectsTrack: (dataset: string, episode: number, lock: string) =>
    fetch(`/api/objects/track?${q({ dataset, episode, lock })}`).then((r) => json<ObjectTrack>(r)),
  jobEvents: (jobId: string) => new EventSource(`/api/jobs/${jobId}/events`),
  cancelJob: (jobId: string) => fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" }).then((r) => json(r)),
  // ---- policy ----
  policies: (lock: string) =>
    fetch(`/api/policy?${q({ lock })}`).then((r) => json<{ lock: string; policies: PolicyInfo[] }>(r)),
  policyTrain: (dataset: string, lock: string, epochs = 200) =>
    fetch("/api/policy/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, lock, epochs }),
    }).then((r) => json<{ job_id: string }>(r)),
  policyEvaluate: (dataset: string, lock: string, mode: PolicyMode, robots?: string[]) =>
    fetch("/api/policy/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, lock, mode, ...(robots ? { robots } : {}) }),
    }).then((r) => json<{ job_id: string }>(r)),
  policyResults: (lock: string) => fetch(`/api/policy/results?${q({ lock })}`).then((r) => json<PolicyResults>(r)),
  policyViewer: (dataset: string, lock: string, episode: number, mode: PolicyMode, robots = "default,lock") =>
    fetch(`/api/policy/viewer?${q({ dataset, lock, episode, mode, robots })}`).then((r) => json<PolicyViewer>(r)),
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
