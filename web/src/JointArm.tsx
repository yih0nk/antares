// A side-view sketch of the SO-100/101 arm with each joint drawn as a dot
// sized and colored by its tracking error, so where the sim is off reads at a
// glance. The scale is absolute (degrees), so a calibrated ghost visibly
// cools down against the default one.

export const ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];

// Error at which a joint is drawn fully hot.
const HOT_DEG = 8;
const STOPS: [number, number, number][] = [
  [63, 214, 184],
  [226, 179, 90],
  [255, 91, 110],
];

export function heat(v: number) {
  const t = Math.min(1, Math.max(0, v / HOT_DEG));
  const [a, b, k] = t < 0.5 ? [STOPS[0], STOPS[1], t * 2] : [STOPS[1], STOPS[2], (t - 0.5) * 2];
  return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * k)).join(",")})`;
}

// Joint positions in the 320x200 sketch, and where each label sits.
const AT: Record<string, { p: [number, number]; dx: number; dy: number; anchor: "start" | "middle" | "end" }> = {
  shoulder_pan: { p: [92, 164], dx: -20, dy: 0, anchor: "end" },
  shoulder_lift: { p: [104, 124], dx: -22, dy: -6, anchor: "end" },
  elbow_flex: { p: [146, 52], dx: 0, dy: -28, anchor: "middle" },
  wrist_flex: { p: [214, 72], dx: -6, dy: -32, anchor: "end" },
  wrist_roll: { p: [252, 84], dx: 0, dy: -32, anchor: "start" },
  gripper: { p: [286, 96], dx: 0, dy: 32, anchor: "middle" },
};
const LINKS: [string, string][] = [
  ["shoulder_pan", "shoulder_lift"],
  ["shoulder_lift", "elbow_flex"],
  ["elbow_flex", "wrist_flex"],
  ["wrist_flex", "wrist_roll"],
  ["wrist_roll", "gripper"],
];

type Props = {
  errors: Record<string, number>;
  selected: string;
  onSelect: (joint: string) => void;
};

export function JointArm({ errors, selected, onSelect }: Props) {
  return (
    <div className="joint-arm">
      <svg viewBox="0 0 320 200" role="img" aria-label="Arm sketch with each joint colored by its tracking error">
        <rect x="66" y="176" width="52" height="12" rx="3" className="arm-base" />
        {LINKS.map(([a, b]) => (
          <line key={a} x1={AT[a].p[0]} y1={AT[a].p[1]} x2={AT[b].p[0]} y2={AT[b].p[1]} className="arm-link" />
        ))}
        <line x1={286} y1={96} x2={312} y2={88} className="arm-finger" />
        <line x1={286} y1={96} x2={312} y2={108} className="arm-finger" />
        {ARM_JOINTS.map((j) => {
          const e = errors[j];
          if (e == null) return null;
          const { p, dx, dy, anchor } = AT[j];
          const r = 7 + Math.min(1, e / HOT_DEG) * 9;
          return (
            <g
              key={j}
              className={`arm-joint ${j === selected ? "sel" : ""}`}
              role="button"
              tabIndex={0}
              aria-label={`${j}: ${e.toFixed(1)} degrees`}
              aria-pressed={j === selected}
              onClick={() => onSelect(j)}
              onKeyDown={(ev) => {
                if (ev.key === "Enter" || ev.key === " ") {
                  ev.preventDefault();
                  onSelect(j);
                }
              }}
            >
              <circle cx={p[0]} cy={p[1]} r={r} fill={heat(e)} />
              <text x={p[0] + dx} y={p[1] + dy} textAnchor={anchor} className="arm-name">
                {j}
              </text>
              <text x={p[0] + dx} y={p[1] + dy + 12} textAnchor={anchor} className="arm-val">
                {e.toFixed(1)}°
              </text>
            </g>
          );
        })}
      </svg>
      <div className="arm-scale">
        <span>0°</span>
        <i />
        <span>{HOT_DEG}°+</span>
      </div>
    </div>
  );
}
