import { useId, type ReactNode } from "react";
import type { CalibrationResult, NumericParamKey, Params, ParamValue, ProgressEvent } from "./api";
import { Sparkline } from "./Charts";
import { Factors } from "./Factors";
import { deg } from "./format";
import { KIND_HELP, ROLE_HELP, type EpisodeRole, type LockKind, type PartialRun } from "./lockInfo";
import type { SimTone } from "./theme";
import {
  Badge,
  Button,
  CheckboxRow,
  Disclosure,
  Icon,
  InlineNotice,
  LegendItem,
  OverflowMenu,
  ProgressBar,
  Spinner,
  Stat,
} from "./ui";

// The Calibrate tab: how big the sim-to-real gap is, which physics the sim
// runs on, the error per joint, and the calibration run itself.

// ---- the calibration job --------------------------------------------------------

export type Calibration = {
  jobId: string;
  name: string;
  // Evaluations the run was asked for over all its stages; sizes the sparkline.
  total: number;
  perJoint: boolean;
  status: "running" | "done" | "error" | "cancelled";
  progress: ProgressEvent[];
  best: number[];
  result?: CalibrationResult;
  error?: string;
};

// CMA-ES budget for a viewer run; also sizes the sparkline's x axis.
export const MAX_EVALUATIONS = 150;
// Budget of the optional per-joint stage (the API's default).
export const PER_JOINT_EVALUATIONS = 600;

const STAGE_LABEL: Record<ProgressEvent["stage"], string> = {
  shared: "Fitting shared parameters",
  per_joint: "Fitting each joint separately",
  command_map: "Fitting the sensor map",
};

// ---- (a) the gap -------------------------------------------------------------------

export function GapStat({
  gap,
  stockGap,
  episode,
  role,
  tone,
  live,
  busy,
}: {
  gap: number | null;
  stockGap: number | null;
  episode: number | null;
  role: EpisodeRole | null;
  tone: SimTone;
  live: boolean;
  busy: boolean;
}) {
  let delta: { glyph: string; text: ReactNode; tone: "good" | "critical" } | null = null;
  if (tone === "sim" && gap != null && stockGap != null && isFinite(gap) && isFinite(stockGap)) {
    const better = gap <= stockGap;
    delta = better
      ? { glyph: "▼", text: `from ${deg(stockGap)} on the stock model`, tone: "good" }
      : { glyph: "▲", text: `worse than ${deg(stockGap)} on the stock model`, tone: "critical" };
  }
  return (
    <section className="section" aria-label="Sim-to-real gap">
      <Stat
        label="Sim-to-real gap"
        value={deg(gap)}
        delta={delta}
        caption={
          <>
            RMS joint error · episode {episode ?? "--"}
            {role && (
              <>
                {" · "}
                <Badge tone={role === "train" ? "warn" : "muted"} glyph={role === "train" ? "!" : undefined} title={ROLE_HELP[role]}>
                  {role}
                </Badge>
              </>
            )}
          </>
        }
        extra={
          live ? (
            <div className="stat-live">
              <Spinner /> Live: the ghost replays the best fit so far
            </div>
          ) : busy ? (
            <div className="stat-live">
              <Spinner /> Replaying with the selected physics…
            </div>
          ) : null
        }
      />
    </section>
  );
}

// ---- (b) the physics picker ------------------------------------------------------------

export type PhysicsRow = {
  name: string;
  label: string;
  // Full name and creation time, for the row's tooltip.
  tooltip: string;
  kinds: LockKind[];
  when: string;
  heldOut: number | null;
  partial: PartialRun | null;
  // The older lock this one repeats exactly (same parameters and error).
  repeatOf: { name: string; label: string } | null;
  exportUrl: string;
};

const VISIBLE_LOCKS = 4;

export function PhysicsPicker({
  rows,
  stockHeldOut,
  live,
  selected,
  onSelect,
  running,
  onDelete,
  showAll,
  onShowAll,
  stockValue,
  liveValue,
  liveMeta,
}: {
  rows: PhysicsRow[];
  stockHeldOut: number | null;
  // The live fit's row: shown while calibrating or while the ghost replays it.
  live: boolean;
  selected: string;
  onSelect: (name: string) => void;
  running: boolean;
  onDelete: (name: string) => void;
  showAll: boolean;
  onShowAll: (v: boolean) => void;
  stockValue: string;
  liveValue: string;
  liveMeta: ReactNode;
}) {
  const name = useId();
  const shown = showAll ? rows : rows.filter((r, i) => i < VISIBLE_LOCKS || r.name === selected);
  return (
    <section className="section">
      <fieldset className="physics">
        <legend className="section-title">Sim physics</legend>
        <div className="section-hint">What the sim ghost replays the real commands with.</div>
        {running && (
          <InlineNotice tone="info">Locked while calibrating: the ghost shows the live fit. Stop the run to pick other physics.</InlineNotice>
        )}
        <div className="phys-list">
          <PhysicsRowView
            group={name}
            value={STOCK}
            checked={selected === STOCK}
            onSelect={onSelect}
            title="Stock model"
            meta={<span className="muted">MJCF defaults</span>}
            value2={stockValue}
            caption={stockHeldOut != null ? "held out" : ""}
            tone="stock"
            disabled={running}
          />
          {live && (
            <PhysicsRowView
              group={name}
              value={LIVE_ID}
              checked={selected === LIVE_ID}
              onSelect={onSelect}
              title={
                <>
                  Live fit {running && <Spinner />}
                </>
              }
              meta={liveMeta}
              value2={liveValue}
              caption="training"
              tone="sim"
            />
          )}
          {shown.map((r) => (
            <PhysicsRowView
              key={r.name}
              group={name}
              value={r.name}
              checked={selected === r.name}
              onSelect={onSelect}
              title={r.label}
              fullName={r.tooltip}
              meta={
                <>
                  {/* "shared" is the default, and a kind the name already says is noise. */}
                  {r.kinds
                    .filter((k) => k !== "shared" && !r.label.toLowerCase().replace(/[-_ ]/g, "").includes(k.replace(/[-_ ]/g, "")))
                    .map((k) => (
                    <Badge key={k} title={KIND_HELP[k]}>
                      {k}
                    </Badge>
                  ))}
                  {r.partial && (
                    <Badge tone="warn" glyph="◐" title={`Stopped after ${r.partial.n} of the ${r.partial.total} evaluations it asked for: the best fit so far was saved.`}>
                      partial · {r.partial.n}/{r.partial.total}
                    </Badge>
                  )}
                  {r.repeatOf && (
                    <Badge tone="muted" title={`Same parameters and held-out error as ${r.repeatOf.name}: runs are deterministic (seed 0), so the same options reproduce the same lock.`}>
                      = {r.repeatOf.label}
                    </Badge>
                  )}
                  {r.when && <span className="muted">{r.when}</span>}
                </>
              }
              value2={deg(r.heldOut)}
              caption={r.heldOut != null ? "held out" : ""}
              tone="sim"
              disabled={running}
              menu={
                <OverflowMenu
                  label={`Actions for ${r.label}`}
                  header={<span className="mono-name">{r.name}</span>}
                  items={[
                    { label: "Download MJCF", href: r.exportUrl, download: `${r.name}.zip`, icon: <Icon.download /> },
                    { label: "Delete", danger: true, onSelect: () => onDelete(r.name), icon: <Icon.close /> },
                  ]}
                />
              }
            />
          ))}
        </div>
        {rows.length > VISIBLE_LOCKS && (
          <Button variant="ghost" size="sm" onClick={() => onShowAll(!showAll)} className="show-all">
            {showAll ? "Show fewer" : `Show all (${rows.length})`}
          </Button>
        )}
        {!rows.length && <div className="section-hint">No saved locks for this dataset yet. Calibrate to make one.</div>}
      </fieldset>
    </section>
  );
}

export const STOCK = "default";
export const LIVE_ID = "live";

function PhysicsRowView({
  group,
  value,
  checked,
  onSelect,
  title,
  fullName,
  meta,
  value2,
  caption,
  tone,
  menu,
  disabled,
}: {
  group: string;
  value: string;
  checked: boolean;
  onSelect: (v: string) => void;
  title: ReactNode;
  fullName?: string;
  meta: ReactNode;
  value2: string;
  caption: string;
  tone: SimTone;
  menu?: ReactNode;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className={`phys-row ${checked ? "on" : ""} ${disabled ? "disabled" : ""}`}>
      <label className="phys-main" title={fullName}>
        <input
          type="radio"
          name={group}
          value={value}
          checked={checked}
          onChange={() => onSelect(value)}
          className="radio"
          disabled={disabled}
          aria-labelledby={`${id}-name ${id}-value ${id}-caption`}
          aria-describedby={`${id}-meta`}
        />
        <span className={`phys-swatch tone-${tone}`} aria-hidden="true" />
        <span className="phys-name" id={`${id}-name`}>
          {title}
        </span>
        <span className="phys-value" id={`${id}-value`}>
          {value2}
        </span>
        <span className="phys-meta" id={`${id}-meta`}>
          {meta}
        </span>
        <span className="phys-caption" id={`${id}-caption`}>
          {caption}
        </span>
      </label>
      {menu ?? <span />}
    </div>
  );
}

// ---- (c) per-joint bars ----------------------------------------------------------------

export function JointBars({
  jointNames,
  scored,
  stock,
  selected,
  tone,
  physicsLabel,
  joint,
  onJoint,
}: {
  jointNames: string[];
  scored: Set<string>;
  stock: Record<string, number> | undefined;
  selected: Record<string, number> | undefined;
  tone: SimTone;
  physicsLabel: string;
  joint: string;
  onJoint: (j: string) => void;
}) {
  const calibrated = tone === "sim";
  const max = Math.max(1e-6, ...Object.values(stock ?? {}), ...Object.values(selected ?? {}));
  return (
    <section className="section">
      <div className="section-head">
        <h2 className="section-title">Per joint</h2>
        <span className="legend-row">
          {calibrated && <LegendItem k={<span className="key bar-key tone-stock" />}>stock model</LegendItem>}
          <LegendItem k={<span className={`key bar-key tone-${tone}`} />}>{calibrated ? "selected" : "stock model"}</LegendItem>
        </span>
      </div>
      <div className="section-hint">
        RMS error on this episode{calibrated ? `, stock model vs ${physicsLabel}` : ""}. Click a joint to trace it below.
      </div>
      <div className="bars">
        {jointNames.map((j) => {
          const s = stock?.[j];
          const g = calibrated ? selected?.[j] : s;
          const isScored = scored.has(j);
          const on = j === joint;
          return (
            <button
              key={j}
              type="button"
              className={`bar-row ${on ? "on" : ""} ${isScored ? "" : "unscored"}`}
              aria-pressed={on}
              onClick={() => onJoint(j)}
              title={
                isScored
                  ? calibrated
                    ? `${j}: stock model ${deg(s)}, ${physicsLabel} ${deg(g)}`
                    : `${j}: stock model ${deg(s)}`
                  : "Not scored: gripper contact isn't simulated"
              }
            >
              <span className="bar-name">{j}</span>
              <span className={`bar-track ${calibrated ? "two" : ""}`}>
                {calibrated && s != null && <span className="bar tone-stock" style={{ width: `${(s / max) * 100}%` }} />}
                {g != null && <span className={`bar tone-${tone}`} style={{ width: `${(g / max) * 100}%` }} />}
              </span>
              <span className="bar-val">{isScored ? deg(g, 1) : "not scored"}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

// ---- (d) the calibration run -------------------------------------------------------------

export type CalibOptions = {
  perJoint: boolean;
  fitGravity: boolean;
  fitDeadband: boolean;
  // null: the built-in factors; otherwise the shared-stage names to fit.
  factors: string[] | null;
};

export function CalibrateSection({
  calib,
  running,
  blocker,
  onStart,
  onStop,
  opts,
  onOpts,
  savedLabel,
  partial,
  onDismissError,
}: {
  calib: Calibration | null;
  running: boolean;
  // Why Calibrate cannot start yet, said next to the button.
  blocker: string | null;
  onStart: () => void;
  onStop: () => void;
  opts: CalibOptions;
  onOpts: (o: CalibOptions) => void;
  savedLabel: string | null;
  partial: PartialRun | null;
  onDismissError: () => void;
}) {
  const last = calib?.progress[calib.progress.length - 1];
  const changed = [opts.perJoint, opts.fitGravity, opts.fitDeadband, opts.factors != null].filter(Boolean).length;
  return (
    <section className="section">
      <Button variant="primary" block busy={running} disabled={!!blocker} onClick={onStart}>
        {running ? "Calibrating…" : "Calibrate"}
      </Button>
      {blocker && !running && <div className="section-hint">{blocker}</div>}
      <div className="section-hint">Runs are deterministic (seed 0): the same options reproduce the same lock.</div>

      {calib && (calib.status === "running" || calib.status === "cancelled") && (
        <div className="run">
          <div className="run-head">
            <span>{calib.status === "cancelled" ? "Stopping after this generation" : last ? STAGE_LABEL[last.stage] : "Starting"}</span>
            <span className="num">
              eval {last?.evaluation ?? 0} / {calib.total}
            </span>
          </div>
          <ProgressBar value={(last?.evaluation ?? 0) / calib.total} label="Calibration progress" />
          <div className="run-head">
            <span className="muted">best so far (training episodes)</span>
            <span className="num">{deg(last?.best_score)}</span>
          </div>
          <Sparkline
            values={calib.best.filter((v) => isFinite(v))}
            total={calib.total}
            baseline={calib.best[0]}
            stageAt={calib.perJoint ? MAX_EVALUATIONS : null}
            label="Best training error so far"
          />
          <Button variant="secondary" block busy={calib.status === "cancelled"} onClick={onStop}>
            {calib.status === "cancelled" ? "Stopping…" : "Stop and save best so far"}
          </Button>
        </div>
      )}

      {calib?.status === "done" && calib.result && (
        <div className="run">
          <div className="run-head">
            <span>
              <Icon.check /> Saved {savedLabel ?? calib.name}
            </span>
            {partial && (
              <Badge tone="warn" glyph="◐" title="Stopped before its budget: the best fit so far was saved.">
                partial · {partial.n}/{partial.total}
              </Badge>
            )}
          </div>
          <dl className="result">
            <div>
              <dt>Held-out (not used in the fit)</dt>
              <dd className="num">
                {deg(calib.result.baseline_eval_score)} → <b>{deg(calib.result.eval_score)}</b>
              </dd>
            </div>
            <div>
              <dt>Training</dt>
              <dd className="num">
                {deg(calib.result.baseline_train_score)} → {deg(calib.result.train_score)}
              </dd>
            </div>
          </dl>
          <Sparkline
            values={calib.best.filter((v) => isFinite(v))}
            total={calib.total}
            baseline={calib.best[0]}
            stageAt={calib.perJoint ? MAX_EVALUATIONS : null}
            label="Best training error over the run"
          />
        </div>
      )}

      {calib?.status === "error" && (
        <InlineNotice tone="error" title="Calibration failed" onDismiss={onDismissError}>
          {calib.error}. Check that <code>antares serve</code> is still running, then press Calibrate to try again.
        </InlineNotice>
      )}

      <Disclosure summary="Options" meta={changed ? `${changed} changed` : "defaults"}>
        {running && <div className="section-hint">Options are locked while a calibration runs.</div>}
        <CheckboxRow
          checked={opts.perJoint}
          disabled={running}
          onChange={(v) => onOpts({ ...opts, perJoint: v })}
          label="Fit each joint separately"
          aside={`+${PER_JOINT_EVALUATIONS} evals`}
          helper="After the shared fit, refine one value per joint. About a minute longer."
        />
        <CheckboxRow
          checked={opts.fitGravity}
          disabled={running}
          onChange={(v) => onOpts({ ...opts, fitGravity: v })}
          label="Fit gravity sag"
          helper="Also fit how much the arm sags under its own weight."
        />
        <CheckboxRow
          checked={opts.fitDeadband}
          disabled={running}
          onChange={(v) => onOpts({ ...opts, fitDeadband: v })}
          label="Fit servo deadband"
          helper="Also fit the servo's dead zone and gear backlash."
        />
        <Factors selected={opts.factors} onChange={(f) => onOpts({ ...opts, factors: f })} disabled={running} />
      </Disclosure>
    </section>
  );
}

// ---- (e) parameters --------------------------------------------------------------------------

const PARAM_KEYS: NumericParamKey[] = ["kp", "kv", "damping", "armature", "frictionloss", "latency_frames", "forcerange", "gravity_scale", "deadband"];
// Terms a lock omits at their defaults; the row shows only when the lock carries one.
const OPTIONAL_DEFAULTS: Partial<Record<NumericParamKey, number>> = { gravity_scale: 1, deadband: 0 };
// The STS3215 servo model's fields, written only by a --servo fit; shown when either side has them.
const SERVO_KEYS: NumericParamKey[] = ["servo_gain", "p_coefficient", "vin", "kt", "resistance", "max_velocity"];
// With a servo model these hold its linearized PD equivalent, not fitted values.
const LINEARIZED_KEYS: ReadonlySet<NumericParamKey> = new Set<NumericParamKey>(["kp", "kv", "forcerange"]);
const servoOf = (p: Params) => p.servo ?? "position";

function fmtScalar(k: NumericParamKey, v: number) {
  if (k === "latency_frames") return `${Math.round(v)} fr`;
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

// A per-joint value shows as its range across the six joints.
function fmtParam(k: NumericParamKey, v: ParamValue) {
  if (!Array.isArray(v)) return fmtScalar(k, v);
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  return lo === hi ? fmtScalar(k, lo) : `${fmtScalar(k, lo)}–${fmtScalar(k, hi)}`;
}

function sameParam(a: ParamValue, b: ParamValue) {
  const av = Array.isArray(a) ? a : [a];
  const bv = Array.isArray(b) ? b : [b];
  const n = Math.max(av.length, bv.length);
  for (let i = 0; i < n; i++) {
    if (Math.abs(av[i % av.length] - bv[i % bv.length]) > 1e-9) return false;
  }
  return true;
}

export function ParamsDisclosure({ stock, shown, label, fullName }: { stock: Params | null; shown: Params | null; label: string; fullName: string | null }) {
  if (!stock || !shown) return null;
  return (
    <section className="section">
      <Disclosure summary="Parameters" meta={`stock vs ${label}`}>
        {fullName && <div className="section-hint mono-name">{fullName}</div>}
        <table className="data-table params">
          <thead>
            <tr>
              <th scope="col">
                <span className="sr-only">parameter</span>
              </th>
              <th scope="col">stock</th>
              <th scope="col">selected</th>
            </tr>
          </thead>
          <tbody>
            {(servoOf(stock) !== "position" || servoOf(shown) !== "position") && (
              <tr>
                <th scope="row">servo</th>
                <td>{servoOf(stock)}</td>
                <Cell changed={servoOf(shown) !== servoOf(stock)}>{servoOf(shown)}</Cell>
              </tr>
            )}
            {PARAM_KEYS.map((k) => {
              const optional = OPTIONAL_DEFAULTS[k];
              if (optional != null && stock[k] == null && shown[k] == null) return null;
              const d = stock[k] ?? optional;
              const g = shown[k] ?? optional;
              if (d == null || g == null) return null; // older locks have no forcerange
              // a servo fit's kp/kv/forcerange are its linearized equivalent, not fitted
              const derived = LINEARIZED_KEYS.has(k) && servoOf(shown) !== "position";
              return (
                <tr key={k}>
                  <th scope="row" title={derived ? "PD equivalent of the servo model (derived, not fitted)" : undefined}>
                    {k}
                    {derived && <span className="muted"> (PD equiv.)</span>}
                  </th>
                  <td>{fmtParam(k, d)}</td>
                  <Cell changed={!sameParam(g, d)}>{fmtParam(k, g)}</Cell>
                </tr>
              );
            })}
            {SERVO_KEYS.map((k) => {
              const d = stock[k];
              const g = shown[k];
              if (d == null && g == null) return null; // position-actuator fits have none
              return (
                <tr key={k}>
                  <th scope="row">{k}</th>
                  <td>{d == null ? "–" : fmtParam(k, d)}</td>
                  <Cell changed={!(d != null && g != null && sameParam(g, d))}>{g == null ? "–" : fmtParam(k, g)}</Cell>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="section-hint">
          <span className="changed-mark" aria-hidden="true">•</span> marks a value that differs from the stock model.
        </div>
      </Disclosure>
    </section>
  );
}

function Cell({ changed, children }: { changed: boolean; children: ReactNode }) {
  return (
    <td className={changed ? "changed" : ""}>
      {changed && (
        <span className="changed-mark" aria-label="changed">
          •{" "}
        </span>
      )}
      {children}
    </td>
  );
}
