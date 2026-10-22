import { useEffect, useState } from "react";
import { api, type CheckResponse, type DatasetInfo } from "./api";
import { Badge, Button, CheckboxRow, InlineNotice, Select } from "./ui";
import { deg } from "./format";

// Run `antares check` for one lock from the fleet view: replay a recording
// through the locked sim and say whether the arm still matches it. The
// result is appended to the lock's history, so the row's Last check and
// Trend update when it lands.
//
// A check only means something on a *new* recording of the same arm, so the
// picker defaults to another dataset simulated on the lock's model; with
// none, it falls back to the fitted recording and warns that this passes by
// construction. A recording on another arm model is marked and warned about:
// it measures how two arms differ, not drift, and it would skew the trend.

type Props = {
  lock: string; // what the check is run against: the lock name or @robot
  lockDataset: string | null;
  lockRobot: string; // the sim model the lock was fitted on
  onChecked: () => void;
};

const VERDICT: Record<string, { tone: "good" | "critical" | "warn"; glyph: string; label: string }> = {
  ok: { tone: "good", glyph: "✓", label: "ok" },
  drifted: { tone: "critical", glyph: "!", label: "drifted" },
  insufficient_data: { tone: "warn", glyph: "?", label: "not enough data" },
};

export function CheckPanel({ lock, lockDataset, lockRobot, onChecked }: Props) {
  const [datasets, setDatasets] = useState<DatasetInfo[] | null>(null);
  const [dataset, setDataset] = useState("");
  const [explain, setExplain] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<CheckResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.datasets().then(
      (ds) => {
        setDatasets(ds);
        const fresh = ds.find((d) => d.id !== lockDataset && d.robot === lockRobot);
        setDataset((cur) => cur || fresh?.id || lockDataset || ds[0]?.id || "");
      },
      () => setDatasets([]),
    );
  }, [lockDataset, lockRobot]);

  const run = async () => {
    if (!dataset || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.check(dataset, lock, explain);
      setResult(r);
      onChecked();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const ownData = !!dataset && dataset === lockDataset;
  const picked = datasets?.find((d) => d.id === dataset);
  const otherArm = !!picked && !ownData && !!picked.robot && picked.robot !== lockRobot;
  const v = result ? VERDICT[result.verdict] ?? VERDICT.insufficient_data : null;

  return (
    <div className="check-panel" onClick={(e) => e.stopPropagation()}>
      <span className="section-label">run a check</span>
      <p className="section-hint">
        Replays a recording through this lock and compares its error with the lock's own. It passes within{" "}
        1.5× of the lock's held-out error. Add a new recording of the arm with Add dataset in the Viewer, then pick it here.
      </p>
      <div className="check-controls">
        <Select
          label="Recording"
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
          disabled={busy || !datasets?.length}
        >
          {(datasets ?? []).map((d) => (
            <option key={d.id} value={d.id}>
              {d.id}
              {d.id === lockDataset ? " (fitted on)" : d.robot && d.robot !== lockRobot ? " (other arm model)" : ""}
            </option>
          ))}
        </Select>
        <Button variant="primary" busy={busy} onClick={run} disabled={!dataset}>
          {busy ? "Checking…" : "Run check"}
        </Button>
      </div>
      <CheckboxRow
        checked={explain}
        onChange={setExplain}
        disabled={busy}
        label="Name what changed if it drifted"
        helper="Refits the sim on the new recording to find the parameter that moved. Adds about a minute."
      />
      {datasets && datasets.length === 0 && (
        <InlineNotice tone="warn">No datasets on this server yet. Add one in the Viewer first.</InlineNotice>
      )}
      {ownData && !result && (
        <InlineNotice tone="warn">
          This is the recording the lock was fitted on, so the check will pass by construction. Use a new recording to test for drift.
        </InlineNotice>
      )}
      {otherArm && !result && (
        <InlineNotice tone="warn">
          This recording is simulated on {picked!.robot}, not {lockRobot}: the check would measure how two arms differ, not drift,
          and it would land in this lock's trend.
        </InlineNotice>
      )}
      {error && (
        <InlineNotice tone="error" title="Check failed" onDismiss={() => setError(null)}>
          {error}
        </InlineNotice>
      )}
      {result && v && (
        <div className="check-result" role="status">
          <div className="check-verdict">
            <Badge tone={v.tone} glyph={v.glyph}>
              {v.label}
            </Badge>
            {result.score != null && (
              <span className="mono">
                {deg(result.score)} <span className="muted">against a limit of {deg(result.limit)}</span>
              </span>
            )}
          </div>
          <div className="section-hint">
            {result.verdict === "insufficient_data"
              ? `Only ${result.n_episodes} usable episode${result.n_episodes === 1 ? "" : "s"}; a check needs at least ${result.min_episodes}.`
              : `${result.n_episodes} episodes of ${result.dataset} · lock reference ${deg(result.reference)} · worst joint ${result.worst_joint ?? "--"}`}
            {ownData && " · on the lock's own recording"}
          </div>
          {result.diagnosis && <div className="section-hint">What changed: {result.diagnosis.summary}</div>}
          {result.diagnosis_error && <div className="section-hint">Could not name what changed: {result.diagnosis_error}</div>}
          {result.history_error && (
            <InlineNotice tone="warn">The verdict stands but was not saved to the history: {result.history_error}</InlineNotice>
          )}
        </div>
      )}
    </div>
  );
}
