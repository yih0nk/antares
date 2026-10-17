// Small SVG charts for the panel and timeline. They size to their container
// width through viewBox + preserveAspectRatio="none" and keep strokes crisp
// with vector-effect.

import { useMemo } from "react";

type TraceProps = {
  real: number[];
  sim: number[];
  frame: number;
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

export function JointTrace({ real, sim, frame, onSeek }: TraceProps) {
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

  return (
    <svg
      className="trace"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      onPointerDown={seek}
      onPointerMove={seek}
    >
      <path d={paths.real} className="trace-real" vectorEffect="non-scaling-stroke" />
      <path d={paths.sim} className="trace-sim" vectorEffect="non-scaling-stroke" />
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
  onSeek?: (frame: number) => void;
};

const RH = 40;

// Where the sim disagrees with the arm: the residual as an area, detected
// events shaded by kind, grasp windows as faint bands. Shares the trace's x
// axis so the two strips line up.
export function ResidualStrip({ sum, events, grasps, frames, frame, onSeek }: ResidualProps) {
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
    >
      {grasps.map(([s, e], i) => (
        <rect key={`g${i}`} x={x(s)} width={Math.max(1, x(e) - x(s))} y={0} height={RH} className="res-grasp" />
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
      <path d={area} className="res-area" />
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

const GOOD = [0x5b, 0x8c, 0xff]; // --sim
const MID = [0x4a, 0x52, 0x66]; // neutral between
const BAD = [0xff, 0x8a, 0x5b]; // --warm

function mix(a: number[], b: number[], t: number) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}

// Blue (good) through gray to warm (bad), positioned on a log scale over the
// matrix's own range so a 6° default and a 0.7° lock both get a readable cell.
function cellColor(v: number, lo: number, hi: number) {
  const l = (x: number) => Math.log(Math.max(1e-3, x));
  const t = hi > lo ? Math.min(1, Math.max(0, (l(v) - l(lo)) / (l(hi) - l(lo)))) : 0.5;
  const c = t < 0.5 ? mix(GOOD, MID, t * 2) : mix(MID, BAD, (t - 0.5) * 2);
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, 0.55)`;
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
            dataset <span className="muted">↓</span> · lock <span className="muted">→</span>
          </th>
          {cols.map((c) => (
            <th key={c} title={c === "default" ? "MJCF defaults, uncalibrated" : `lock ${c}, fitted on ${short(home[c] ?? "?")}`}>
              {c === "default" ? "default sim" : c.replace(/_/g, "_​")}
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
};

// Best-so-far error over evaluations, on a fixed x range so the line grows.
export function Sparkline({ values, total, baseline, stageAt }: SparkProps) {
  const W = 280;
  const H = 64;
  if (!values.length) return <svg className="spark" viewBox={`0 0 ${W} ${H}`} />;
  const hi = Math.max(baseline ?? 0, ...values);
  const lo = 0;
  const n = Math.max(total, values.length);
  const y = (v: number) => H - 2 - ((v - lo) / (hi - lo || 1)) * (H - 4);
  const d = values.map((v, i) => `${i ? "L" : "M"}${((i / Math.max(1, n - 1)) * W).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const lastX = ((values.length - 1) / Math.max(1, n - 1)) * W;
  const stageX = stageAt != null && stageAt > 0 && stageAt < n ? (stageAt / Math.max(1, n - 1)) * W : null;
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
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
