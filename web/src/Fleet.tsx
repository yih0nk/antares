import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Dashboard, type DashboardRobot, type LockHistory, type ParamValue } from "./api";
import { HistorySpark, type HistoryPoint } from "./Charts";

// One screen for a fleet: every saved lock as a row with its calibration
// quality, its last `antares check` and the trend of those checks. Rows
// expand to the check history and the per-joint error of the last check.

type Status = "ok" | "drifted" | "insufficient" | "never";

const STATUS_LABEL: Record<Status, string> = {
  ok: "ok",
  drifted: "drifted",
  insufficient: "insufficient data",
  never: "never checked",
};

function verdictStatus(verdict: string | null | undefined): Status {
  if (!verdict) return "never";
  if (verdict === "ok") return "ok";
  if (verdict === "drifted") return "drifted";
  return "insufficient";
}

const statusOf = (r: DashboardRobot) => verdictStatus(r.history.latest?.verdict);

const short = (id: string) => id.split("/").pop() ?? id;
const deg = (v: number | null | undefined, digits = 2) => (v == null || !isFinite(v) ? "--" : `${v.toFixed(digits)}°`);
const pct = (v: number | null | undefined) => (v == null || !isFinite(v) ? "--" : `${Math.round(v * 100)} %`);
const date = (iso: string | null | undefined) => {
  if (!iso) return "--";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
};

function median(xs: number[]): number | null {
  const s = xs.filter((x) => isFinite(x)).sort((a, b) => a - b);
  if (!s.length) return null;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// A per-joint lock has one latency per joint; the table shows the range.
function latencyFrames(v: ParamValue | null): { lo: number; hi: number } | null {
  if (v == null) return null;
  const xs = (Array.isArray(v) ? v : [v]).filter((x) => isFinite(x));
  return xs.length ? { lo: Math.min(...xs), hi: Math.max(...xs) } : null;
}

function fmtLatency(r: DashboardRobot): string {
  const f = latencyFrames(r.latency_frames);
  if (!f) return "--";
  const frames = f.lo === f.hi ? `${f.hi}` : `${f.lo}–${f.hi}`;
  if (!r.fps) return `${frames} fr`;
  const ms = (n: number) => Math.round((n * 1000) / r.fps!);
  return `${frames} fr · ${f.lo === f.hi ? ms(f.hi) : `${ms(f.lo)}–${ms(f.hi)}`} ms`;
}

// Slope of the check scores, and the fit's r² when the trend has one.
function fmtTrend(r: DashboardRobot): { slope: string; r2: string | null } | null {
  const t = r.history.trend;
  if (!t || t.slope_deg_per_day == null || t.n < 2) return null;
  const slope = t.slope_deg_per_day;
  return { slope: `${slope >= 0 ? "+" : ""}${slope.toFixed(3)}°/day`, r2: t.r2 == null ? null : `r² ${t.r2.toFixed(2)}` };
}

function Trend({ robot }: { robot: DashboardRobot }) {
  const t = fmtTrend(robot);
  if (!t) return <>n/a</>;
  return (
    <>
      {t.slope}
      {t.r2 && (
        <>
          {" "}
          <span className="faint">{t.r2}</span>
        </>
      )}
    </>
  );
}

type SortKey = "robot" | "dataset" | "calibrated" | "error" | "gap" | "latency" | "check" | "trend" | "worst" | "pinned";

const COLUMNS: { key: SortKey; label: string; title: string }[] = [
  { key: "robot", label: "robot", title: "lock name" },
  { key: "dataset", label: "dataset", title: "the LeRobot dataset the lock was fitted on" },
  { key: "calibrated", label: "calibrated", title: "when the lock was written" },
  { key: "error", label: "default → calibrated", title: "held-out tracking RMSE, default sim → this lock" },
  { key: "gap", label: "gap closed", title: "fraction of the default sim's error the lock removed" },
  { key: "latency", label: "latency", title: "fitted command latency (frames, and ms at the dataset's fps)" },
  { key: "check", label: "last check", title: "the newest `antares check` of this lock: score vs limit" },
  { key: "trend", label: "trend", title: "slope of the check scores over time, with the fit's r²" },
  { key: "worst", label: "worst joint", title: "the body joint with the largest held-out error after calibration" },
  { key: "pinned", label: "pinned", title: "parameters a seeded calibration held fixed" },
];

const STATUS_RANK: Record<Status, number> = { drifted: 0, insufficient: 1, ok: 2, never: 3 };

function sortValue(r: DashboardRobot, k: SortKey): number | string | null {
  switch (k) {
    case "robot":
      return r.name;
    case "dataset":
      return r.dataset ?? "";
    case "calibrated":
      return r.created_at ? Date.parse(r.created_at) : null;
    case "error":
      return r.eval_score;
    case "gap":
      return r.improvement;
    case "latency":
      return latencyFrames(r.latency_frames)?.hi ?? null;
    case "check":
      // Worst status first, then the most recent check.
      return r.history.latest ? STATUS_RANK[statusOf(r)] * 1e13 - Date.parse(r.history.latest.at) : null;
    case "trend":
      return r.history.trend?.slope_deg_per_day ?? null;
    case "worst":
      return r.worst_joint_deg;
    case "pinned":
      return r.pinned?.length ?? null;
  }
}

function compare(a: DashboardRobot, b: DashboardRobot, key: SortKey, dir: 1 | -1) {
  const va = sortValue(a, key);
  const vb = sortValue(b, key);
  if (va == null && vb == null) return a.name.localeCompare(b.name);
  if (va == null) return 1; // missing values sink whatever the direction
  if (vb == null) return -1;
  const c = typeof va === "string" || typeof vb === "string" ? String(va).localeCompare(String(vb)) : va - vb;
  return c * dir || a.name.localeCompare(b.name);
}

type HistoryState = LockHistory | "loading" | "error";

// `initialOpen` is a lock name to expand on mount (from the `#fleet/<lock>` hash).
export function FleetView({ initialOpen = null }: { initialOpen?: string | null }) {
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "check", dir: 1 });
  const [open, setOpen] = useState<string | null>(initialOpen);
  const [histories, setHistories] = useState<Record<string, HistoryState>>({});

  const load = useCallback(() => {
    setBusy(true);
    api.dashboard().then(
      (d) => {
        setData(d);
        setError(null);
        setBusy(false);
      },
      (e) => {
        setError(e.status === 404 ? "This server has no /api/dashboard route; update antares." : `Could not load the fleet: ${e.message}`);
        setBusy(false);
      },
    );
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  const robots = data?.robots ?? [];
  const showPinned = robots.some((r) => r.pinned && r.pinned.length > 0);
  const columns = showPinned ? COLUMNS : COLUMNS.filter((c) => c.key !== "pinned");

  const stats = useMemo(() => {
    const statuses = robots.map(statusOf);
    const checked = statuses.filter((s) => s !== "never").length;
    return {
      robots: robots.length,
      inFleet: robots.filter((r) => r.in_fleet).length,
      medianGap: median(robots.map((r) => r.improvement ?? NaN)),
      drifted: statuses.filter((s) => s === "drifted").length,
      checked,
      never: statuses.filter((s) => s === "never").length,
    };
  }, [robots]);

  const sorted = useMemo(() => [...robots].sort((a, b) => compare(a, b, sort.key, sort.dir)), [robots, sort]);

  const sortBy = (key: SortKey) =>
    setSort((cur) => (cur.key === key ? { key, dir: cur.dir === 1 ? -1 : 1 } : { key, dir: key === "robot" || key === "dataset" ? 1 : -1 }));

  const fetchHistory = useCallback((name: string) => {
    setHistories((h) => ({ ...h, [name]: "loading" }));
    api.history(name).then(
      (h) => setHistories((cur) => ({ ...cur, [name]: h })),
      () => setHistories((cur) => ({ ...cur, [name]: "error" })),
    );
  }, []);
  useEffect(() => {
    if (initialOpen) fetchHistory(initialOpen);
  }, [initialOpen, fetchHistory]);

  const toggle = (name: string) => {
    const next = open === name ? null : name;
    setOpen(next);
    window.history.replaceState(null, "", next ? `#fleet/${encodeURIComponent(next)}` : "#fleet");
    if (!histories[name]) fetchHistory(name);
  };

  return (
    <main className="fleet">
      <div className="fleet-head">
        <div>
          <span className="label">fleet</span>
          <div className="small muted">
            {data
              ? `${robots.length} lock${robots.length === 1 ? "" : "s"} in the lock directory`
              : busy
                ? "loading…"
                : "no data"}
            {data?.fleet && (
              <>
                {" · "}
                <span title={`fleet scan ${date(data.fleet.created_at)}: ${data.fleet.n_robots} robots, median gap closed ${pct(data.fleet.median_improvement)}`}>
                  last fleet scan {date(data.fleet.created_at)} · {data.fleet.n_robots} robots
                </span>
              </>
            )}
            {data && ` · as of ${new Date(data.generated_at).toLocaleTimeString()}`}
          </div>
        </div>
        <button className="cta secondary small-cta" onClick={load} disabled={busy}>
          {busy ? "refreshing…" : "refresh"}
        </button>
      </div>

      {error && <div className="fleet-error">{error}</div>}

      <div className="tiles">
        <div className="tile">
          <span className="label">robots</span>
          <span className="value">{stats.robots}</span>
          <span className="small muted">{stats.inFleet ? `${stats.inFleet} from the fleet scan` : "saved locks"}</span>
        </div>
        <div className="tile">
          <span className="label">median gap closed</span>
          <span className="value">{pct(stats.medianGap)}</span>
          <span className="small muted">held-out error, default → calibrated</span>
        </div>
        <div className="tile">
          <span className="label">drifted at last check</span>
          <span className={`value ${stats.drifted ? "drifted" : ""}`}>{stats.drifted}</span>
          <span className="small muted">
            {stats.checked ? `of ${stats.checked} checked` : "no robot has been checked yet"}
          </span>
        </div>
        <div className="tile">
          <span className="label">never checked</span>
          <span className={`value ${stats.never ? "never" : ""}`}>{stats.never}</span>
          <span className="small muted">no `antares check` on record</span>
        </div>
      </div>

      {data && !robots.length ? (
        <div className="fleet-empty muted">
          No locks yet. Calibrate an arm in the Viewer, or run <span className="mono">antares fleet scan</span>.
        </div>
      ) : (
        <div className="fleet-table-wrap">
          <table className="fleet-table">
            <thead>
              <tr>
                {columns.map((c) => (
                  <th key={c.key} aria-sort={sort.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
                    <button className={sort.key === c.key ? "on" : ""} onClick={() => sortBy(c.key)} title={c.title}>
                      {c.label}
                      {sort.key === c.key && <span className="sort-dir">{sort.dir === 1 ? "▲" : "▼"}</span>}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => {
                const status = statusOf(r);
                const latest = r.history.latest;
                const isOpen = open === r.name;
                return [
                  <tr
                    key={r.name}
                    className={`robot ${isOpen ? "open" : ""}`}
                    onClick={() => toggle(r.name)}
                    aria-expanded={isOpen}
                  >
                    <td>
                      <div className="robot-name">
                        <span className="mono" title={r.name}>
                          {r.name}
                        </span>
                        {r.in_fleet && (
                          <span className="chip tiny" title="calibrated by `antares fleet scan`">
                            fleet
                          </span>
                        )}
                        {r.space === "per_joint" && (
                          <span className="chip tiny" title="per-joint parameters">
                            per joint
                          </span>
                        )}
                        {r.drift.exists && (
                          <span
                            className={`chip tiny ${r.drift.verdict && /^drift/i.test(r.drift.verdict) ? "warn" : ""}`}
                            title={r.drift.verdict ?? "a drift scan exists for this lock"}
                          >
                            session drift
                          </span>
                        )}
                      </div>
                    </td>
                    <td title={r.dataset ?? undefined}>{r.dataset ? short(r.dataset) : "--"}</td>
                    <td className="mono nowrap">{date(r.created_at)}</td>
                    <td className="mono nowrap">
                      <span className="muted">{deg(r.baseline_eval_score)}</span> → {deg(r.eval_score)}
                    </td>
                    <td className="mono nowrap">{pct(r.improvement)}</td>
                    <td className="mono nowrap">{fmtLatency(r)}</td>
                    <td>
                      <div className="check-cell">
                        <span className={`chip verdict ${status}`}>{STATUS_LABEL[status]}</span>
                        {latest && (
                          <span className="small muted mono" title={`${latest.n_episodes} episodes from ${latest.dataset}`}>
                            {date(latest.at)} · {deg(latest.score)}
                            {latest.limit != null && <> / {deg(latest.limit)}</>}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="mono nowrap">
                      <Trend robot={r} />
                    </td>
                    <td className="nowrap">
                      {r.worst_joint ? (
                        <>
                          {r.worst_joint} <span className="mono muted">{deg(r.worst_joint_deg)}</span>
                        </>
                      ) : (
                        "--"
                      )}
                    </td>
                    {showPinned && <td className="mono">{r.pinned && r.pinned.length ? r.pinned.join(", ") : "--"}</td>}
                  </tr>,
                  isOpen && (
                    <tr key={`${r.name}:detail`} className="fleet-detail">
                      <td colSpan={columns.length}>
                        <RobotDetail robot={r} history={histories[r.name]} />
                      </td>
                    </tr>
                  ),
                ];
              })}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}

function RobotDetail({ robot: r, history }: { robot: DashboardRobot; history: HistoryState | undefined }) {
  const entries = history && typeof history !== "string" ? history.entries : [];
  const points: HistoryPoint[] = entries
    .filter((e) => e.score != null && isFinite(e.score))
    .map((e) => ({ at: e.at, score: e.score!, limit: e.limit, status: verdictStatus(e.verdict) }));
  const trend = (history && typeof history !== "string" && history.trend) || r.history.trend;
  const latest = entries.length ? entries[entries.length - 1] : r.history.latest;
  const status = verdictStatus(latest?.verdict);

  const reference = r.per_joint_after ?? {};
  const joints = Object.keys(reference).length ? Object.keys(reference) : Object.keys(latest?.per_joint ?? {});
  const scale = Math.max(
    1e-6,
    ...joints.map((j) => Math.max(reference[j] ?? 0, latest?.per_joint?.[j] ?? 0)),
  );

  return (
    <div className="detail-grid">
      <section>
        <span className="label">check history</span>
        {history === "loading" ? (
          <span className="small muted">loading…</span>
        ) : history === "error" ? (
          <span className="small muted">history unavailable</span>
        ) : !points.length ? (
          <span className="small muted">
            never checked · <span className="mono">antares check &lt;dataset&gt; --lock {r.name}</span>
          </span>
        ) : (
          <>
            <HistorySpark points={points} reference={r.eval_score} />
            <div className="small muted">
              {points.length} check{points.length === 1 ? "" : "s"} · {date(points[0].at)} → {date(points[points.length - 1].at)}
              {trend && trend.n_drifted > 0 && <> · {trend.n_drifted} drifted</>}
              {" · "}trend <Trend robot={{ ...r, history: { latest: r.history.latest, trend } }} />
            </div>
            <div className="small muted legend-inline">
              <span>
                <i className="swatch spark-ref" /> lock reference {deg(r.eval_score)}
              </span>
              {latest?.limit != null && (
                <span>
                  <i className="swatch warm" /> limit {deg(latest.limit)}
                </span>
              )}
            </div>
          </>
        )}
      </section>

      <section>
        <span className="label">{latest ? "per joint · last check vs lock" : "per joint · after calibration"}</span>
        <div className="bars">
          {joints.map((j) => {
            const ref = reference[j];
            const chk = latest?.per_joint?.[j];
            // With a check on record the bar is the check and the lock is a
            // tick; without one the bar is the lock's own held-out error.
            return (
              <div className={`bar-row static ${j === "gripper" ? "unscored" : ""}`} key={j}>
                <span>{j}</span>
                <div className={`bar-track ${latest ? "with-ref" : ""}`}>
                  {latest ? (
                    chk != null && <div className={`bar check ${status}`} style={{ width: `${(chk / scale) * 100}%` }} />
                  ) : (
                    ref != null && <div className="bar ghost" style={{ width: `${(ref / scale) * 100}%` }} />
                  )}
                  {latest && ref != null && (
                    <div className="bar-ref" style={{ left: `${(ref / scale) * 100}%` }} title={`lock ${deg(ref)}`} />
                  )}
                </div>
                <span className="bar-val" title={latest && ref != null ? `lock ${deg(ref)}` : undefined}>
                  {deg(latest ? chk : ref)}
                </span>
              </div>
            );
          })}
        </div>
        <div className="small muted legend-inline">
          {latest ? (
            <>
              <span>
                <i className={`swatch ${status}`} /> last check
              </span>
              <span>
                <i className="swatch tick" /> lock
              </span>
            </>
          ) : (
            <span>
              <i className="swatch sim" /> lock, held out
            </span>
          )}
        </div>
      </section>

      {(r.drift.exists || (r.pinned && r.pinned.length > 0)) && (
        <section className="detail-notes small">
          {r.drift.exists && (
            <div>
              <span className="label">session drift</span>
              <div>{r.drift.verdict ?? "scan on disk, no verdict recorded"}</div>
            </div>
          )}
          {r.pinned && r.pinned.length > 0 && (
            <div>
              <span className="label">pinned</span>
              <div className="mono">{r.pinned.join(", ")}</div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
