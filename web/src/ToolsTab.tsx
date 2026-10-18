import type { Compensation } from "./api";
import { deg, plural } from "./format";
import { Button, ButtonLink, CheckboxRow, Icon, Spinner } from "./ui";

// The Tools tab: lag compensation, the cross-robot matrix, and export.

export function ToolsTab({
  showComp,
  onShowComp,
  comp,
  compBusy,
  compBlocked,
  physicsLabel,
  onOpenCross,
  nLoaded,
  nLocks,
  exportLock,
  exportLabel,
  exportUrl,
}: {
  showComp: boolean;
  onShowComp: (v: boolean) => void;
  comp: Compensation | null;
  compBusy: boolean;
  // Why lag compensation can't run right now (the live fit has no saved lock).
  compBlocked: string | null;
  physicsLabel: string;
  onOpenCross: () => void;
  nLoaded: number;
  nLocks: number;
  exportLock: string | null;
  exportLabel: string;
  exportUrl: string | null;
}) {
  return (
    <>
      <section className="section">
        <h2 className="section-title">Lag compensation</h2>
        <CheckboxRow
          checked={showComp}
          disabled={!!compBlocked}
          onChange={onShowComp}
          label={
            <>
              Show lag-compensated sim {compBusy && <Spinner />}
            </>
          }
          helper="Iterative learning control finds commands that make the sim follow the real arm despite servo latency. Adds a magenta arm."
        />
        {compBlocked && <div className="section-hint">{compBlocked}</div>}
        {showComp && !compBlocked && (
          <>
            <dl className="result three">
              {(["raw", "shift", "ilc"] as const).map((k) => (
                <div key={k}>
                  <dt>{k === "raw" ? "Raw" : k === "shift" ? "Shifted" : "ILC"}</dt>
                  <dd className="num">{compBusy ? "…" : k === "ilc" && comp ? <b>{deg(comp.rms[k])}</b> : deg(comp?.rms[k])}</dd>
                </div>
              ))}
            </dl>
            <div className="section-hint">
              RMS tracking error of the sim ({physicsLabel}) on the real trajectory: raw commands, commands shifted by the fitted latency, and
              iterative learning control{comp && ` (${comp.ilc_iterations} iterations)`}. Sim-validated only: judged in MuJoCo, no hardware
              driven.
            </div>
          </>
        )}
      </section>

      <section className="section">
        <h2 className="section-title">Every robot is different</h2>
        <div className="section-hint">Replay every saved lock on every loaded dataset's held-out episodes to see how badly a fit transfers to another arm.</div>
        <Button onClick={onOpenCross} disabled={!nLocks}>
          Compare locks across robots
        </Button>
        <div className="section-hint">
          {nLocks
            ? `${plural(nLoaded, "dataset")} loaded × ${plural(nLocks, "lock")}${nLoaded < 2 ? " · load another dataset (top bar) to add a row" : ""}`
            : "Needs at least one saved lock: calibrate first."}
        </div>
      </section>

      <section className="section">
        <h2 className="section-title">Export</h2>
        {exportLock && exportUrl ? (
          <>
            <ButtonLink href={exportUrl} download={`${exportLock}.zip`} icon={<Icon.download />}>
              Download MJCF
            </ButtonLink>
            <div className="section-hint" title={exportLock}>
              {exportLabel}: a standalone MuJoCo model with the fitted physics (model, sidecar, scene and meshes).
            </div>
          </>
        ) : (
          <div className="section-hint">Pick calibrated physics in the Calibrate tab to export it as a standalone MuJoCo model.</div>
        )}
      </section>
    </>
  );
}
