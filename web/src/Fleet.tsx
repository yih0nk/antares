import { useCallback, useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { api, type Dashboard, type DashboardRobot, type LockHistory, type ParamValue } from "./api";
import { HistorySpark, type HistoryPoint } from "./Charts";
import { CheckPanel } from "./CheckPanel";
import { Badge, Button, Icon, InlineNotice, LegendItem, LineKey, type Tone } from "./ui";

// One screen for a fleet: every saved lock as a row with its calibration
// quality, its last `antares check` and the trend of those checks. Rows
// expand to the check history and the per-joint error of the last check.
//
// When the lock directory has a registry (locks/registry.json), locks are
// grouped under their physical robot: one header row per robot showing the
// active lock's numbers and a "v3 of 3" badge, expandable to the lineage.
// Locks no robot claims stay as flat rows under "unassigned".

type Status = "ok" | "drifted" | "insufficient" | "never";

const STATUS_LABEL: Record<Status, string> = {
  ok: "ok",
  drifted: "drifted",
  insufficient: "insufficient data",
  never: "never checked",
};

// Status is never colour alone: each verdict has a glyph and a word.
const STATUS_BADGE: Record<Status, { tone: Tone; glyph: string }> = {
  ok: { tone: "good", glyph: "✓" },
  drifted: { tone: "critical", glyph: "▲" },
  insufficient: { tone: "warn", glyph: "◐" },
  never: { tone: "muted", glyph: "○" },
};

// Enter or Space on a focused table row toggles it, like a click.
const rowKeys = (toggle: () => void) => (e: KeyboardEvent<HTMLTableRowElement>) => {
  if (e.target !== e.currentTarget || (e.key !== "Enter" && e.key !== " ")) return;
  e.preventDefault();
  toggle();
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
  { key: "robot", label: "robot", title: "registered robot id, or the lock name" },
  { key: "dataset", label: "dataset", title: "the LeRobot dataset the lock was fitted on" },
  { key: "calibrated", label: "calibrated", title: "when the lock was written" },
  { key: "error", label: "stock → calibrated", title: "held-out tracking RMSE, stock model → this lock" },
  { key: "gap", label: "gap closed", title: "fraction of the stock model's error the lock removed" },
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
      return r.robot_id ?? r.name;
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

// A registered robot: its lineage rows and the one the header row shows
// (the active lock, or the newest version when none is active).
type RobotGroup = { id: string; rows: DashboardRobot[]; head: DashboardRobot; nVersions: number };

function groupByRobot(robots: DashboardRobot[]): { groups: RobotGroup[]; loose: DashboardRobot[]; grouped: boolean } {
  const grouped = robots.some((r) => r.robot_id);
  if (!grouped) return { groups: [], loose: robots, grouped };
  const byId = new Map<string, DashboardRobot[]>();
  const loose: DashboardRobot[] = [];
  for (const r of robots) {
    if (!r.robot_id) loose.push(r);
    else byId.set(r.robot_id, [...(byId.get(r.robot_id) ?? []), r]);
  }
  const groups: RobotGroup[] = [];
  for (const [id, rows] of byId) {
    rows.sort((a, b) => (b.version ?? 0) - (a.version ?? 0)); // newest version first
    const head = rows.find((r) => r.is_active) ?? rows[0];
    groups.push({ id, rows, head, nVersions: Math.max(head.n_versions ?? 0, rows.length) });
  }
  return { groups, loose, grouped };
}

type HistoryState = LockHistory | "loading" | "error";

// `initialOpen` is a lock name to expand on mount (from the `#fleet/<lock>` hash).
export function FleetView({ initialOpen = null }: { initialOpen?: string | null }) {
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "check", dir: 1 });
  const [open, setOpen] = useState<string | null>(initialOpen);
  const [openRobots, setOpenRobots] = useState<Set<string>>(new Set());
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

  const { groups, loose, grouped } = useMemo(() => groupByRobot(robots), [robots]);

  // A deep link to a lock inside a lineage opens its robot too.
  useEffect(() => {
    if (!initialOpen) return;
    const g = groups.find((g) => g.rows.some((r) => r.name === initialOpen));
    if (g) setOpenRobots((cur) => (cur.has(g.id) ? cur : new Set([...cur, g.id])));
  }, [initialOpen, groups]);

  const stats = useMemo(() => {
    // With a registry, a robot is judged by its active lock; without one, every lock is a robot.
    const judged = grouped ? [...groups.map((g) => g.head), ...loose] : robots;
    const statuses = judged.map(statusOf);
    const checked = statuses.filter((s) => s !== "never").length;
    return {
      robots: judged.length,
      registered: groups.length,
      locks: robots.length,
      unassigned: loose.length,
      inFleet: robots.filter((r) => r.in_fleet).length,
      medianGap: median(judged.map((r) => r.improvement ?? NaN)),
      drifted: statuses.filter((s) => s === "drifted").length,
      checked,
      never: statuses.filter((s) => s === "never").length,
    };
  }, [robots, groups, loose, grouped]);

  const sortedGroups = useMemo(
    () =>
      [...groups].sort((a, b) =>
        sort.key === "robot" ? a.id.localeCompare(b.id) * sort.dir : compare(a.head, b.head, sort.key, sort.dir),
      ),
    [groups, sort],
  );
  const sortedLoose = useMemo(() => [...loose].sort((a, b) => compare(a, b, sort.key, sort.dir)), [loose, sort]);

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

  const toggleRobot = (id: string) =>
    setOpenRobots((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  // One lock as a table row: a flat (unassigned) lock or a lineage member.
  const lockRow = (r: DashboardRobot, kind: "flat" | "lineage") => {
    const isOpen = open === r.name;
    return [
      <tr
        key={r.name}
        className={`robot ${kind} ${isOpen ? "open" : ""}`}
        onClick={() => toggle(r.name)}
        onKeyDown={rowKeys(() => toggle(r.name))}
        tabIndex={0}
        aria-expanded={isOpen}
      >
        <td>
          <div className="robot-name">
            {kind === "lineage" && r.version != null && (
              <Badge tone={r.is_active ? "good" : "muted"} glyph={r.is_active ? "●" : undefined} title={r.is_active ? "the active lock of this robot" : "an older calibration of this robot"}>
                v{r.version}
                {r.is_active && " · active"}
              </Badge>
            )}
            <span className="lock-name" title={r.name}>
              {r.name}
            </span>
            <LockChips r={r} />
          </div>
        </td>
        <LockCells r={r} showPinned={showPinned} />
      </tr>,
      isOpen && (
        <tr key={`${r.name}:detail`} className="fleet-detail">
          <td colSpan={columns.length}>
            <RobotDetail robot={r} history={histories[r.name]} onChecked={() => { load(); fetchHistory(r.name); }} />
          </td>
        </tr>
      ),
    ];
  };

  // A registered robot: the active lock's numbers, expandable to its lineage.
  const robotRows = (g: RobotGroup) => {
    const expanded = openRobots.has(g.id);
    const head = g.head;
    const version = head.version != null ? `v${head.version} of ${g.nVersions}` : `${g.nVersions} lock${g.nVersions === 1 ? "" : "s"}`;
    return [
      <tr
        key={`robot:${g.id}`}
        className={`robot group-head ${expanded ? "expanded" : ""}`}
        onClick={() => toggleRobot(g.id)}
        onKeyDown={rowKeys(() => toggleRobot(g.id))}
        tabIndex={0}
        aria-expanded={expanded}
      >
        <td>
          <div className="robot-name">
            <span className="caret" aria-hidden="true">
              <Icon.chevron />
            </span>
            <span className="robot-id" title={`robot ${g.id}; active lock ${head.name}`}>
              {g.id}
            </span>
            <Badge tone="good" glyph="●" title={head.is_active ? `active lock ${head.name}, version ${head.version} of ${g.nVersions}` : `no active lock; showing ${head.name}`}>
              {version}
            </Badge>
            {!head.is_active && (
              <Badge tone="warn" glyph="!" title="the registry has no active lock for this robot">
                no active lock
              </Badge>
            )}
            <LockChips r={head} />
          </div>
        </td>
        <LockCells r={head} showPinned={showPinned} />
      </tr>,
      ...(expanded ? g.rows.flatMap((r) => lockRow(r, "lineage")) : []),
    ];
  };

  return (
    <main className="fleet">
      <div className="fleet-head">
        <div>
          <h1 className="section-title">Fleet</h1>
          <div className="section-hint">
            {data
              ? `${robots.length} lock${robots.length === 1 ? "" : "s"} in the lock directory`
              : busy
                ? "loading…"
                : "no data"}
            {data && grouped && (
              <>
                {" · "}
                <span title="locks/registry.json: one id per physical robot, its lock lineage and the active lock">
                  {stats.registered} registered robot{stats.registered === 1 ? "" : "s"}
                  {stats.unassigned ? `, ${stats.unassigned} unassigned lock${stats.unassigned === 1 ? "" : "s"}` : ""}
                </span>
              </>
            )}
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
        <Button busy={busy} onClick={load}>
          {busy ? "Refreshing…" : "Refresh"}
        </Button>
      </div>

      {error && (
        <InlineNotice tone="error" title="Could not load the fleet" onDismiss={() => setError(null)}>
          {error}
        </InlineNotice>
      )}

      <div className="tiles">
        <div className="tile">
          <span className="section-label">robots</span>
          <span className="value">{stats.robots}</span>
          <span className="caption">
            {grouped
              ? `${stats.registered} registered · ${stats.locks} lock${stats.locks === 1 ? "" : "s"}`
              : stats.inFleet
                ? `${stats.inFleet} from the fleet scan`
                : "saved locks"}
          </span>
        </div>
        <div className="tile">
          <span className="section-label">median gap closed</span>
          <span className="value">{pct(stats.medianGap)}</span>
          <span className="caption">held-out error, stock model → calibrated{grouped ? " (active locks)" : ""}</span>
        </div>
        <div className="tile">
          <span className="section-label">drifted at last check</span>
          <span className={`value ${stats.drifted ? "drifted" : ""}`}>
            {stats.drifted > 0 && <Icon.warn />}
            {stats.drifted}
          </span>
          <span className="caption">{stats.checked ? `of ${stats.checked} checked` : "no robot has been checked yet"}</span>
        </div>
        <div className="tile">
          <span className="section-label">never checked</span>
          <span className="value">{stats.never}</span>
          <span className="caption">no `antares check` on record</span>
        </div>
      </div>

      {data && !robots.length ? (
        <div className="fleet-empty">
          No locks yet. Calibrate an arm in the Viewer, or run <code>antares fleet scan</code>.
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
                      {sort.key === c.key && (
                        <span className="sort-dir" aria-hidden="true">
                          {sort.dir === 1 ? "▲" : "▼"}
                        </span>
                      )}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedGroups.flatMap(robotRows)}
              {grouped && sortedLoose.length > 0 && (
                <tr className="group-divider" key="divider:unassigned">
                  <td colSpan={columns.length} title="locks no registered robot claims; `antares robots register <id> --lock <name>` assigns one">
                    unassigned · {sortedLoose.length} lock{sortedLoose.length === 1 ? "" : "s"}
                  </td>
                </tr>
              )}
              {sortedLoose.flatMap((r) => lockRow(r, "flat"))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}

// The badges that describe a lock, after its name.
function LockChips({ r }: { r: DashboardRobot }) {
  return (
    <>
      {r.in_fleet && <Badge title="calibrated by `antares fleet scan`">fleet</Badge>}
      {r.space === "per_joint" && <Badge title="one value per joint for each parameter">per-joint</Badge>}
      {r.drift.exists &&
        (r.drift.verdict && /^drift/i.test(r.drift.verdict) ? (
          <Badge tone="warn" glyph="!" title={r.drift.verdict}>
            session drift
          </Badge>
        ) : (
          <Badge title={r.drift.verdict ?? "a drift scan exists for this lock"}>session drift</Badge>
        ))}
    </>
  );
}

// Every column after the name, for one lock.
function LockCells({ r, showPinned }: { r: DashboardRobot; showPinned: boolean }) {
  const status = statusOf(r);
  const latest = r.history.latest;
  return (
    <>
      <td title={r.dataset ?? undefined}>{r.dataset ? short(r.dataset) : "--"}</td>
      <td className="nowrap">{date(r.created_at)}</td>
      <td className="nowrap">
        <span className="faint">{deg(r.baseline_eval_score)}</span> → {deg(r.eval_score)}
      </td>
      <td className="nowrap">{pct(r.improvement)}</td>
      <td className="nowrap">{fmtLatency(r)}</td>
      <td>
        <div className="check-cell">
          <Badge tone={STATUS_BADGE[status].tone} glyph={STATUS_BADGE[status].glyph}>
            {STATUS_LABEL[status]}
          </Badge>
          {latest && (
            <span className="when" title={`${latest.n_episodes} episodes from ${latest.dataset}`}>
              {date(latest.at)} · {deg(latest.score)}
              {latest.limit != null && <> / {deg(latest.limit)}</>}
            </span>
          )}
        </div>
      </td>
      <td className="nowrap">
        <Trend robot={r} />
      </td>
      <td className="nowrap">
        {r.worst_joint ? (
          <>
            {r.worst_joint} <span className="faint">{deg(r.worst_joint_deg)}</span>
          </>
        ) : (
          "--"
        )}
      </td>
      {showPinned && <td>{r.pinned && r.pinned.length ? r.pinned.join(", ") : "--"}</td>}
    </>
  );
}

function RobotDetail({ robot: r, history, onChecked }: { robot: DashboardRobot; history: HistoryState | undefined; onChecked: () => void }) {
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
  const checkRef = r.robot_id && r.is_active ? `@${r.robot_id}` : r.name;

  return (
    <div className="detail-grid">
      <section>
        <span className="section-label">check history</span>
        {history === "loading" ? (
          <span className="section-hint">Loading…</span>
        ) : history === "error" ? (
          <span className="section-hint">History unavailable.</span>
        ) : !points.length ? (
          <span className="section-hint">
            Never checked. Run one below, or from a shell: <code>antares check &lt;dataset&gt; --lock {checkRef}</code>
          </span>
        ) : (
          <>
            <HistorySpark points={points} reference={r.eval_score} />
            <div className="section-hint">
              {points.length} check{points.length === 1 ? "" : "s"} · {date(points[0].at)} → {date(points[points.length - 1].at)}
              {trend && trend.n_drifted > 0 && <> · {trend.n_drifted} drifted</>}
              {" · "}trend <Trend robot={{ ...r, history: { latest: r.history.latest, trend } }} />
            </div>
            <div className="legend-row">
              <LegendItem k={<LineKey tone="stock" dashed />}>lock reference {deg(r.eval_score)}</LegendItem>
              {latest?.limit != null && <LegendItem k={<LineKey tone="critical" />}>limit {deg(latest.limit)}</LegendItem>}
            </div>
          </>
        )}
        <CheckPanel lock={checkRef} lockDataset={r.dataset} lockRobot={r.robot} onChecked={onChecked} />
      </section>

      <section>
        <span className="section-label">{latest ? "per joint · last check vs lock" : "per joint · after calibration"}</span>
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
                    ref != null && <div className="bar tone-sim" style={{ width: `${(ref / scale) * 100}%` }} />
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
        <div className="legend-row">
          {latest ? (
            <>
              <LegendItem k={<span className={`key status-key ${status}`} />}>last check ({STATUS_LABEL[status]})</LegendItem>
              <LegendItem k={<span className="key tick-key" />}>lock</LegendItem>
            </>
          ) : (
            <LegendItem k={<span className="key bar-key tone-sim" />}>lock, held out</LegendItem>
          )}
        </div>
      </section>

      {(r.drift.exists || (r.pinned && r.pinned.length > 0) || r.robot_id) && (
        <section className="detail-notes section-hint">
          {r.robot_id && (
            <div>
              <span className="section-label">robot</span>
              <div>
                <span>{r.robot_id}</span> · version {r.version ?? "?"} of {r.n_versions ?? "?"}
                {r.is_active ? " · active" : " · not active"}
              </div>
            </div>
          )}
          {r.drift.exists && (
            <div>
              <span className="section-label">session drift</span>
              <div>{r.drift.verdict ?? "scan on disk, no verdict recorded"}</div>
            </div>
          )}
          {r.pinned && r.pinned.length > 0 && (
            <div>
              <span className="section-label">pinned</span>
              <div>{r.pinned.join(", ")}</div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
