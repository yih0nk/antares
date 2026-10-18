import { useEffect, useRef, useState, type DragEvent } from "react";
import { ApiError, api, type DatasetInfo, type Verdict } from "./api";

// "+ add dataset": a popover under the top bar with two ways in. A Hub id is
// checked first (meta/info.json only, no download) and the verdict shown
// before anything is loaded; a zip of a LeRobot dataset directory is
// uploaded with progress and validated server-side. Either way the new
// dataset lands in the picker and is selected; the existing overlay then
// covers the actual load.

type Props = {
  disabled?: boolean;
  // Called with the API's summary of the new dataset (loaded or not).
  onAdded: (info: DatasetInfo) => void;
};

type Upload = {
  name: string;
  fraction: number;
  status: "uploading" | "done" | "error";
  message?: string;
  verdict?: Verdict | null;
};

export function VerdictLines({ verdict }: { verdict: Verdict }) {
  return (
    <ul className="verdict">
      {verdict.reasons.map((r) => (
        <li key={r} className="bad">
          {r}
        </li>
      ))}
      {verdict.warnings.map((w) => (
        <li key={w} className="warn">
          {w}
        </li>
      ))}
      {verdict.ok && !verdict.warnings.length && <li className="good">looks good</li>}
    </ul>
  );
}

const fmtVerdictHead = (v: Verdict) =>
  `${v.robot_type || "unknown robot"}${v.robot_family ? ` (${v.robot_family})` : ""} · ${v.n_episodes} episodes @ ${v.fps} fps`;

export function AddDataset({ disabled, onAdded }: Props) {
  const [open, setOpen] = useState(false);
  const [hubId, setHubId] = useState("");
  const [checking, setChecking] = useState(false);
  const [checked, setChecked] = useState<{ id: string; verdict: Verdict; robot: string } | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [upload, setUpload] = useState<Upload | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  // Close on Escape or a click outside.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onDown = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onDown);
    };
  }, [open]);

  const check = async () => {
    const id = hubId.trim();
    if (!id) return;
    setChecking(true);
    setCheckError(null);
    setChecked(null);
    try {
      const v = await api.validateDataset(id);
      setChecked({ id, verdict: v, robot: v.robot });
    } catch (e) {
      setCheckError((e as Error).message);
    } finally {
      setChecking(false);
    }
  };

  const load = () => {
    if (!checked?.verdict.ok) return;
    const v = checked.verdict;
    onAdded({
      id: checked.id,
      loaded: false,
      source: "hub",
      warnings: v.warnings,
      robot_family: v.robot_family,
      robot: checked.robot,
      robot_type: v.robot_type,
      fps: v.fps,
      n_episodes: v.n_episodes,
    });
    setOpen(false);
    setHubId("");
    setChecked(null);
  };

  const send = async (file: File) => {
    if (!file) return;
    setUpload({ name: file.name, fraction: 0, status: "uploading" });
    try {
      const info = await api.uploadDataset(file, (f) => setUpload((u) => (u ? { ...u, fraction: f } : u)));
      setUpload({ name: file.name, fraction: 1, status: "done", verdict: info.verdict });
      onAdded(info);
      // Leave the verdict on screen for a moment, then close.
      window.setTimeout(() => {
        setOpen(false);
        setUpload(null);
      }, info.verdict.warnings.length ? 2500 : 600);
    } catch (e) {
      const err = e as ApiError;
      setUpload({ name: file.name, fraction: 1, status: "error", message: err.message, verdict: err.verdict ?? null });
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) void send(file);
  };

  return (
    <div className="add-dataset" ref={box}>
      <button className="add-btn" onClick={() => setOpen((o) => !o)} disabled={disabled} title="Bring your own dataset">
        + add dataset
      </button>
      {open && (
        <div className="popover" role="dialog" aria-label="Add a dataset">
          <div className="label">Hugging Face Hub</div>
          <form
            className="hub-row"
            onSubmit={(e) => {
              e.preventDefault();
              void check();
            }}
          >
            <input
              className="picker hub-id"
              placeholder="owner/dataset_name"
              value={hubId}
              onChange={(e) => {
                setHubId(e.target.value);
                setChecked(null);
                setCheckError(null);
              }}
              spellCheck={false}
              autoFocus
            />
            <button type="submit" className="cta secondary small-cta" disabled={!hubId.trim() || checking}>
              {checking ? "Checking…" : "Check"}
            </button>
            <button type="button" className="cta small-cta" disabled={!checked?.verdict.ok} onClick={load}>
              Load
            </button>
          </form>
          {checkError && <div className="bad small">{checkError}</div>}
          {checked && (
            <div className="small">
              <div className="muted">{fmtVerdictHead(checked.verdict)}</div>
              <VerdictLines verdict={checked.verdict} />
            </div>
          )}
          <div className="muted small">
            Any public LeRobot dataset of one SO-100/101 arm: six joints, 30 or 50 fps, at least 5 episodes. Checked
            from its info.json before the download starts.
          </div>

          <div className="label">Upload a zip</div>
          <div
            className={`dropzone ${dragOver ? "over" : ""} ${upload?.status === "uploading" ? "busy" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => upload?.status !== "uploading" && fileInput.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && fileInput.current?.click()}
          >
            <input
              ref={fileInput}
              type="file"
              accept=".zip,application/zip"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void send(f);
                e.target.value = "";
              }}
            />
            {upload ? (
              <div className="upload-state">
                <div className="row">
                  <span className="mono small">{upload.name}</span>
                  <span className="mono small">
                    {upload.status === "uploading" ? `${Math.round(upload.fraction * 100)}%` : upload.status}
                  </span>
                </div>
                <div className="progress">
                  <span style={{ width: `${Math.round(upload.fraction * 100)}%` }} />
                </div>
                {upload.status === "uploading" && upload.fraction >= 1 && (
                  <div className="muted small">Extracting, validating and reading the parquet files…</div>
                )}
                {upload.message && <div className="bad small">{upload.message}</div>}
                {upload.verdict && <VerdictLines verdict={upload.verdict} />}
              </div>
            ) : (
              <>
                <div>Drop a .zip of a LeRobot dataset directory, or click to choose</div>
                <div className="muted small">meta/info.json + data/**/*.parquet; videos are ignored. Up to 200 MB.</div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
