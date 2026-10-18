import { useEffect, useId, useRef } from "react";
import type { CrossMatrix } from "./api";
import { CrossHeatmap } from "./Charts";
import { short } from "./format";
import { Button, Icon, IconButton, InlineNotice } from "./ui";

// "Compare locks across robots": every saved lock replayed on every loaded
// dataset's held-out episodes, as a heatmap in a modal dialog.

export function CrossModal({
  onClose,
  data,
  busy,
  error,
  onRun,
  datasets,
  locks,
  home,
}: {
  onClose: () => void;
  data: CrossMatrix | null;
  busy: boolean;
  error: string | null;
  onRun: () => void;
  datasets: string[];
  locks: string[];
  home: Record<string, string>;
}) {
  const titleId = useId();
  const dialog = useRef<HTMLDivElement | null>(null);

  // Focus moves into the dialog, Escape closes it, Tab stays inside, and
  // focus returns to whatever opened it.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    dialog.current?.querySelector<HTMLElement>("button")?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key !== "Tab" || !dialog.current) return;
      const els = Array.from(dialog.current.querySelectorAll<HTMLElement>("button:not(:disabled), [href], [tabindex]:not([tabindex='-1'])"));
      if (!els.length) return;
      const first = els[0];
      const last = els[els.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      opener?.focus?.();
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby={titleId} ref={dialog} onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <h2 className="section-title" id={titleId}>
              Compare locks across robots
            </h2>
            <div className="section-hint">
              Each saved lock replayed on each loaded dataset's held-out episodes (20 train / 10 held out, seed 0). RMS joint error in degrees;
              lower is better. The outlined cell is a lock on the robot it was fitted on.
            </div>
          </div>
          <IconButton label="Close" onClick={onClose}>
            <Icon.close />
          </IconButton>
        </div>
        {data ? (
          <div className="cross-wrap">
            <CrossHeatmap rows={data.rows} cols={data.cols} scores={data.scores} perJoint={data.per_joint} home={home} />
          </div>
        ) : (
          <div className="cross-empty section-hint">
            {busy ? `Replaying ${datasets.length * (locks.length + 1)} lock × dataset pairs…` : "Nothing computed yet for this set of datasets and locks."}
          </div>
        )}
        {error && (
          <InlineNotice tone="error" title="Could not score the matrix">
            {error}
          </InlineNotice>
        )}
        <div className="modal-foot">
          <span className="section-hint">
            {datasets.map(short).join(", ")} × {locks.length} lock{locks.length === 1 ? "" : "s"} and the stock model
          </span>
          <Button variant="primary" busy={busy} onClick={onRun} disabled={!datasets.length}>
            {busy ? "Scoring…" : data ? "Run again" : "Run"}
          </Button>
        </div>
      </div>
    </div>
  );
}
