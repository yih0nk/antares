// Small SVG charts for the panel and timeline. They size to their container
// width through viewBox + preserveAspectRatio="none" and keep strokes crisp
// with vector-effect.

import { useId, useMemo } from "react";
import { hexToRgb, palette, type SimTone } from "./theme";

type TraceProps = {
  real: number[];
  sim: number[];
  frame: number;
  // Which physics the sim trace runs on: stock grey or calibrated blue.
  simTone?: SimTone;
  // Frame rate, for the playhead's spoken value ("2.30 s of 10.10 s").
  fps?: number;
  label?: string;
  onSeek?: (frame: number) => void;
};

function path(values: number[], n: number, lo: number, hi: number, w: number, h: number) {
  const span = hi - lo || 1;
  let d = "";
  for (let i = 0; i < values.length; i++) {
    const x = (i / Math.max(1, n - 1)) * w;
    const y = h - ((values[i] - lo) / span) * h;
    d += `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }
  return d;
}

const W = 1000;
const H = 100;

// One joint over the episode: the real arm as a solid line, the sim dashed.
// It doubles as the timeline's scrubber (a slider: click or drag to seek;
// the arrow keys are handled by the viewer's shortcuts).
export function JointTrace({ real, sim, frame, simTone = "sim", fps = 30, label = "Playhead", onSeek }: TraceProps) {
  const n = Math.max(real.length, sim.length);
  // The playhead moves every frame; the paths only change with the data.
  const paths = useMemo(() => {
    const all = real.concat(sim);
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const pad = (hi - lo) * 0.08 || 1;
    return {
      real: path(real, n, lo - pad, hi + pad, W, H),
      sim: path(sim, n, lo - pad, hi + pad, W, H),
    };
  }, [real, sim, n]);
  const x = (frame / Math.max(1, n - 1)) * W;

  const seek = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!onSeek || (e.type === "pointermove" && e.buttons !== 1)) return;
    const r = e.currentTarget.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    onSeek(t * (n - 1));
  };

  const secs = (f: number) => `${(f / Math.max(1, fps)).toFixed(2)} s`;
  return (
    <svg
      className="trace"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      onPointerDown={seek}
      onPointerMove={seek}
      role="slider"
      tabIndex={0}
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={Math.max(0, n - 1)}
      aria-valuenow={Math.round(frame)}
      aria-valuetext={`${secs(frame)} of ${secs(Math.max(0, n - 1))}`}
      aria-keyshortcuts="ArrowLeft ArrowRight Shift+ArrowLeft Shift+ArrowRight"
    >
      <path d={paths.real} className="trace-real" vectorEffect="non-scaling-stroke" />
      <path d={paths.sim} className={`trace-sim tone-${simTone}`} vectorEffect="non-scaling-stroke" />
      <line x1={x} x2={x} y1={0} y2={H} className="trace-head" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

type ResidualProps = {
  // Summed |real - sim| over the scored joints, one value per frame.
  sum: number[];
  // Half-open [start, end) frame spans.
  events: { start: number; end: number; kind: "contact" | "anomaly"; joints: string[]; peak_deg: number }[];
  grasps: number[][];
  frames: number;
  frame: number;
  // The residual's physics: stock grey or calibrated blue.
  simTone?: SimTone;
  onSeek?: (frame: number) => void;
};

const RH = 40;

// Where the sim disagrees with the arm: the residual as an area, detected
// events shaded by kind (contact critical, anomaly warn), gripper-closed
// windows as a neutral hatch. Shares the trace's x axis so the two line up.
export function ResidualStrip({ sum, events, grasps, frames, frame, simTone = "sim", onSeek }: ResidualProps) {
  const hatch = `hatch-${useId().replace(/:/g, "")}`;
  const n = Math.max(frames, sum.length);
  const x = (f: number) => (f / Math.max(1, n - 1)) * W;
  const area = useMemo(() => {
    if (!sum.length) return "";
    const hi = Math.max(1e-6, ...sum);
    const lastX = ((sum.length - 1) / Math.max(1, n - 1)) * W;
    return `${path(sum, n, 0, hi * 1.05, W, RH)}L${lastX.toFixed(2)},${RH}L0,${RH}Z`;
  }, [sum, n]);

  const seek = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!onSeek || (e.type === "pointermove" && e.buttons !== 1)) return;
    const r = e.currentTarget.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    onSeek(t * (n - 1));
  };

  return (
    <svg
      className="residual"
      viewBox={`0 0 ${W} ${RH}`}
      preserveAspectRatio="none"
      onPointerDown={seek}
      onPointerMove={seek}
      role="img"
      aria-label={`Residual |real − sim| over the episode: ${events.filter((e) => e.kind === "contact").length} contact and ${
        events.filter((e) => e.kind === "anomaly").length
      } anomaly spans, ${grasps.length} gripper-closed windows`}
    >
      <defs>
        <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" className="res-hatch-line" />
        </pattern>
      </defs>
      {grasps.map(([s, e], i) => (
        <rect key={`g${i}`} x={x(s)} width={Math.max(1, x(e) - x(s))} y={0} height={RH} className="res-grasp" style={{ fill: `url(#${hatch})` }} />
      ))}
      {events.map((ev, i) => (
        <rect
          key={`e${i}`}
          x={x(ev.start)}
          width={Math.max(1, x(ev.end) - x(ev.start))}
          y={0}
          height={RH}
          className={`res-event ${ev.kind}`}
        >
          <title>
            {ev.kind}: {ev.joints.join(", ")} · peak {ev.peak_deg.toFixed(1)}°
          </title>
        </rect>
      ))}
      <path d={area} className={`res-area tone-${simTone}`} />
      <line x1={x(frame)} x2={x(frame)} y1={0} y2={RH} className="trace-head" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

type CrossProps = {
  rows: string[];
  cols: string[];
  scores: number[][];
  perJoint?: Record<string, number>[][];
  // Which dataset each lock was fitted on; a lock on its own dataset is the
  // diagonal and gets outlined.
  home: Record<string, string>;
};

const GOOD = hexToRgb(palette.good); // low error
const MID = hexToRgb(palette.borderStrong); // neutral between
const BAD = hexToRgb(palette.critical); // high error

function mix(a: number[], b: number[], t: number) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}

// Good (low error) through grey to critical (high), positioned on a log scale
// over the matrix's own range so a 6° default and a 0.7° lock both get a
// readable cell. Every cell also prints its number: colour is never alone.
function cellColor(v: number, lo: number, hi: number) {
  const l = (x: number) => Math.log(Math.max(1e-3, x));
  const t = hi > lo ? Math.min(1, Math.max(0, (l(v) - l(lo)) / (l(hi) - l(lo)))) : 0.5;
  const c = t < 0.5 ? mix(GOOD, MID, t * 2) : mix(MID, BAD, (t - 0.5) * 2);
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, 0.42)`;
}

const short = (id: string) => id.split("/").pop() ?? id;

// Every lock scored on every dataset: a lock transfers poorly to another unit.
export function CrossHeatmap({ rows, cols, scores, perJoint, home }: CrossProps) {
  const flat = scores.flat().filter((v) => isFinite(v));
  const lo = Math.min(...flat);
  const hi = Math.max(...flat);
  return (
    <table className="cross">
      <thead>
        <tr>
          <th className="corner">
            dataset <span className="muted">↓</span> · physics <span className="muted">→</span>
          </th>
          {cols.map((c) => (
            <th key={c} title={c === "default" ? "Stock model: the MJCF defaults, uncalibrated" : `lock ${c}, fitted on ${short(home[c] ?? "?")}`}>
              {c === "default" ? "stock model" : c.replace(/_/g, "_\u200b")}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={r}>
            <th title={r}>{short(r)}</th>
            {cols.map((c, j) => {
              const v = scores[i]?.[j];
              const own = home[c] === r;
              const pj = perJoint?.[i]?.[j];
              const tip = pj
                ? Object.entries(pj)
                    .map(([k, d]) => `${k} ${d.toFixed(2)}°`)
                    .join("\n")
                : undefined;
              return (
                <td
                  key={c}
                  className={own ? "own" : ""}
                  style={{ background: v != null ? cellColor(v, lo, hi) : undefined }}
                  title={`${short(r)} × ${c}${own ? " (its own robot)" : ""}\n${tip ?? ""}`}
                >
                  {v == null ? "--" : `${v.toFixed(2)}°`}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

type SparkProps = {
  values: number[];
  total: number;
  baseline?: number | null;
  // Evaluation index where a second stage starts; drawn as a divider.
  stageAt?: number | null;
  // What the line is, for screen readers.
  label?: string;
};

// Best-so-far error over evaluations, on a fixed x range so the line grows.
export function Sparkline({ values, total, baseline, stageAt, label = "Best score so far" }: SparkProps) {
  const W = 280;
  const H = 64;
  if (!values.length) return <svg className="spark" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}: no evaluations yet`} />;
  const hi = Math.max(baseline ?? 0, ...values);
  const lo = 0;
  const n = Math.max(total, values.length);
  const y = (v: number) => H - 2 - ((v - lo) / (hi - lo || 1)) * (H - 4);
  const d = values.map((v, i) => `${i ? "L" : "M"}${((i / Math.max(1, n - 1)) * W).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const lastX = ((values.length - 1) / Math.max(1, n - 1)) * W;
  const stageX = stageAt != null && stageAt > 0 && stageAt < n ? (stageAt / Math.max(1, n - 1)) * W : null;
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={`${label}: ${values.length} of ${n} evaluations`}>
      {baseline != null && (
        <line x1={0} x2={W} y1={y(baseline)} y2={y(baseline)} className="spark-base" vectorEffect="non-scaling-stroke" />
      )}
      {stageX != null && (
        <line x1={stageX} x2={stageX} y1={0} y2={H} className="spark-stage" vectorEffect="non-scaling-stroke" />
      )}
      <path d={`${d}L${lastX.toFixed(1)},${H}L0,${H}Z`} className="spark-fill" />
      <path d={d} className="spark-line" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export type HistoryPoint = {
  at: string;
  score: number;
  limit: number | null;
  // Verdict class for the marker: ok | drifted | insufficient.
  status: string;
};

type HistorySparkProps = {
  // Check scores in time order (oldest first).
  points: HistoryPoint[];
  // The lock's own held-out score; the dashed line the checks are judged against.
  reference?: number | null;
};

// One lock's check scores over calendar time: the limit as a critical rule, the
// lock's reference dashed, one marker per check coloured by its verdict.
// Uniform scaling (not `none`) keeps the markers round.
export function HistorySpark({ points, reference }: HistorySparkProps) {
  const W = 320;
  const H = 72;
  const PAD = 8;
  if (!points.length) return <svg className="spark history" viewBox={`0 0 ${W} ${H}`} />;
  const ts = points.map((p) => Date.parse(p.at));
  const t0 = Math.min(...ts);
  const t1 = Math.max(...ts);
  const x = (t: number) => (t1 > t0 ? PAD + ((t - t0) / (t1 - t0)) * (W - 2 * PAD) : W / 2);
  const limit = points[points.length - 1].limit;
  const hi = Math.max(reference ?? 0, limit ?? 0, ...points.map((p) => p.score)) * 1.15 || 1;
  const y = (v: number) => H - PAD - (v / hi) * (H - 2 * PAD);
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(ts[i]).toFixed(1)},${y(p.score).toFixed(1)}`).join("");
  return (
    <svg className="spark history" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMinYMid meet" role="img">
      <title>
        {points.length} check{points.length === 1 ? "" : "s"}, {points[0].at.slice(0, 10)} to{" "}
        {points[points.length - 1].at.slice(0, 10)}
      </title>
      {reference != null && (
        <line x1={0} x2={W} y1={y(reference)} y2={y(reference)} className="spark-base" vectorEffect="non-scaling-stroke" />
      )}
      {limit != null && (
        <line x1={0} x2={W} y1={y(limit)} y2={y(limit)} className="spark-limit" vectorEffect="non-scaling-stroke" />
      )}
      {points.length > 1 && <path d={d} className="spark-line" vectorEffect="non-scaling-stroke" />}
      {points.map((p, i) => (
        <circle key={i} cx={x(ts[i])} cy={y(p.score)} r={4} className={`spark-dot ${p.status}`}>
          <title>
            {p.at.slice(0, 10)}: {p.score.toFixed(2)}°{p.limit != null ? ` (limit ${p.limit.toFixed(2)}°)` : ""} ·{" "}
            {p.status}
          </title>
        </circle>
      ))}
    </svg>
  );
}
