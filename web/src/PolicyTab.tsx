import type { PolicyInfo, PolicyMode, PolicyResults, PolicyTrainProgress, PolicyViewer } from "./api";
import { cm, deg } from "./format";
import { Badge, Button, CheckboxRow, EmptyState, InlineNotice, ProgressBar, SegmentedControl, Spinner } from "./ui";

// The Policy tab: train a small policy on a lock's training episodes, judge
// it in the stock and the calibrated sim, and show its ghosts.

export type PolicyJob = {
  jobId: string;
  kind: "train" | "evaluate";
  mode?: PolicyMode;
  status: "running" | "done" | "error";
  progress: PolicyTrainProgress[];
  // Evaluation progress: episodes done / total.
  i: number;
  n: number;
  error?: string;
};

// Epochs a viewer training run asks for (early stopping usually ends it sooner).
export const POLICY_EPOCHS = 200;

const MODE_HELP: Record<PolicyMode, string> = {
  shadow: "Shadow: the policy's commands on the real observations, replayed through each sim.",
  closedloop: "Closed loop: the policy drives each sim from the real start toward the same goal.",
};

const variantName = (name: string) =>
  name === "default" ? "stock model" : name === "lock" ? "calibrated" : name === "perjoint" ? "per-joint lock" : name;

export function PolicyTab({
  lock,
  lockLabel,
  borrowed,
  policy,
  job,
  running,
  ready,
  mode,
  onMode,
  onTrain,
  onEvaluate,
  results,
  showGhosts,
  onShowGhosts,
  episode,
  onEpisode,
  viewer,
  viewerNote,
  viewerBusy,
  onGoCalibrate,
  onDismissError,
}: {
  lock: string | null;
  lockLabel: string;
  // The stock model is selected, so the policy shown is the dataset's newest lock's.
  borrowed: boolean;
  policy: PolicyInfo | null;
  job: PolicyJob | null;
  running: boolean;
  ready: boolean;
  mode: PolicyMode;
  onMode: (m: PolicyMode) => void;
  onTrain: () => void;
  onEvaluate: (m: PolicyMode) => void;
  results: PolicyResults | null;
  showGhosts: boolean;
  onShowGhosts: (v: boolean) => void;
  episode: number | null;
  onEpisode: (ep: number) => void;
  viewer: PolicyViewer | null;
  viewerNote: string | null;
  viewerBusy: boolean;
  onGoCalibrate: () => void;
  onDismissError: () => void;
}) {
  if (!lock) {
    return (
      <EmptyState
        title="No lock to train a policy on"
        actions={
          <Button variant="primary" onClick={onGoCalibrate}>
            Go to Calibrate
          </Button>
        }
      >
        A policy trains on a lock's training episodes and is judged on its held-out ones. Calibrate first to save a lock.
      </EmptyState>
    );
  }
  const busy = job?.status === "running";
  const training = job?.kind === "train" && busy;
  const evaluating = job?.kind === "evaluate" && busy;
  const lastEpoch = job?.progress[job.progress.length - 1];
  const heldOut = policy?.eval_episodes ?? [];
  const result = mode === "shadow" ? results?.shadow ?? null : results?.closedloop ?? null;
  const rows = result ? Object.entries(result.summary.variants) : [];
  const shadowLock = results?.shadow?.summary.variants["lock"] ?? null;
  const shadowDefault = results?.shadow?.summary.variants["default"] ?? null;
  const clLock = results?.closedloop?.summary.variants["lock"] ?? null;
  const clDefault = results?.closedloop?.summary.variants["default"] ?? null;
  const blocker = running ? "Waiting for the calibration to finish." : !ready ? "Waiting for the dataset to load." : null;

  return (
    <>
      <section className="section">
        <div className="section-head">
          <h2 className="section-title">Policy</h2>
          <span className="section-hint" title={lock}>
            saved with {lockLabel}
          </span>
        </div>
        {borrowed && (
          <div className="section-hint">The stock model has no lock of its own, so this uses the dataset's newest lock, {lockLabel}.</div>
        )}
        <div className="section-hint">
          A small state-based policy (ACT without images): the last 2 joint states and the grasp point in, the next 10 commands out. Trained
          on the lock's training episodes, stopped early on its held-out ones. Seconds on a CPU; needs the server's [policy] extra.
        </div>
        <Button variant={policy ? "secondary" : "primary"} block busy={training} disabled={!!blocker || busy} onClick={onTrain}>
          {training ? "Training…" : policy ? "Retrain policy" : "Train policy"}
        </Button>
        {blocker && <div className="section-hint">{blocker}</div>}
        {training && (
          <>
            <ProgressBar value={(lastEpoch?.epoch ?? 0) / POLICY_EPOCHS} label="Training progress" />
            <div className="section-hint num">
              {lastEpoch
                ? `epoch ${lastEpoch.epoch} · held-out mse ${lastEpoch.eval_mse.toFixed(3)} (best ${lastEpoch.best_eval_mse.toFixed(3)})`
                : "starting…"}
            </div>
          </>
        )}
        {job?.status === "error" && (
          <InlineNotice tone="error" title={job.kind === "train" ? "Training failed" : "Evaluation failed"} onDismiss={onDismissError}>
            {job.error}
          </InlineNotice>
        )}
        {policy && (
          <dl className="result">
            <div>
              <dt>Held-out chunk error</dt>
              <dd className="num">
                <b>{deg(policy.metrics.policy.eval.rmse_deg)}</b>
              </dd>
            </div>
            <div>
              <dt>Hold state / repeat command</dt>
              <dd className="num">
                {deg(policy.metrics.baselines.hold_state.eval.rmse_deg)} / {deg(policy.metrics.baselines.repeat_command.eval.rmse_deg)}
              </dd>
            </div>
          </dl>
        )}
        {policy && (
          <div className="section-hint">
            {policy.name} · {policy.train_episodes.length} train / {policy.eval_episodes.length} held-out episodes
          </div>
        )}
      </section>

      {policy && (
        <section className="section">
          <h2 className="section-title">Evaluate in both sims</h2>
          <SegmentedControl<PolicyMode>
            label="Mode"
            value={mode}
            onChange={onMode}
            options={[
              { value: "shadow", label: "Shadow", title: MODE_HELP.shadow, disabled: busy },
              { value: "closedloop", label: "Closed loop", title: MODE_HELP.closedloop, disabled: busy },
            ]}
          />
          <ul className="explain-list">
            <li className={mode === "shadow" ? "on" : ""}>{MODE_HELP.shadow}</li>
            <li className={mode === "closedloop" ? "on" : ""}>{MODE_HELP.closedloop}</li>
          </ul>
          <Button variant="primary" block busy={evaluating} disabled={!!blocker || busy} onClick={() => onEvaluate(mode)}>
            {evaluating ? `Evaluating… ${job!.i}/${job!.n || heldOut.length}` : `Evaluate (${mode === "shadow" ? "shadow" : "closed loop"})`}
          </Button>
          {evaluating && <ProgressBar value={job!.n ? job!.i / job!.n : null} label="Evaluation progress" />}

          {result && (
            <>
              <div className="subscore">
                {mode === "shadow" && shadowLock && shadowDefault ? (
                  <>
                    <span className="num big">{deg(shadowLock.divergence_deg)}</span>
                    <span className="section-hint">
                      between what the stock and the calibrated sim say this policy does (sim vs real: {deg(shadowDefault.sim_vs_real_deg)} stock,{" "}
                      {deg(shadowLock.sim_vs_real_deg)} calibrated)
                    </span>
                  </>
                ) : mode === "closedloop" && clLock && clDefault ? (
                  <>
                    <span className="num big">
                      {clDefault.n_success}/{clDefault.n_judged} → {clLock.n_success}/{clLock.n_judged}
                    </span>
                    <span className="section-hint">
                      held-out episodes where the fingertip closes within 2 cm of the real grasp point: stock model → calibrated sim
                    </span>
                  </>
                ) : null}
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">
                      <span className="sr-only">sim</span>
                    </th>
                    {mode === "shadow" ? (
                      <>
                        <th scope="col" title="RMS of the sim's trajectory against the real arm">
                          vs real
                        </th>
                        <th scope="col" title="RMS between this sim's trajectory and the stock model's">
                          vs stock
                        </th>
                        <th scope="col" title="Fingertip distance from the real grasp point">
                          grasp
                        </th>
                      </>
                    ) : (
                      <>
                        <th scope="col" title="RMS of the approach to the grasp against the real arm">
                          approach
                        </th>
                        <th scope="col" title="Fingertip distance from the real grasp point">
                          grasp
                        </th>
                        <th scope="col" title="Episodes where the fingertip closes within 2 cm of the real grasp point">
                          ≤ 2 cm
                        </th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([name, row]) => (
                    <tr key={name}>
                      <th scope="row">{variantName(name)}</th>
                      {"sim_vs_real_deg" in row ? (
                        <>
                          <td>{deg(row.sim_vs_real_deg)}</td>
                          <td>{"divergence_deg" in row ? deg(row.divergence_deg) : "–"}</td>
                          <td>{cm(row.grasp_error_cm)}</td>
                        </>
                      ) : (
                        <>
                          <td>{deg(row.approach_rms_deg, 1)}</td>
                          <td>{cm(row.grasp_error_cm)}</td>
                          <td>
                            {row.n_success}/{row.n_judged}
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="section-hint">{result.summary.n_episodes} held-out episodes, sim only: nothing was driven on hardware.</div>
            </>
          )}
          {!result && !evaluating && <div className="section-hint">No {mode === "shadow" ? "shadow" : "closed-loop"} evaluation saved yet.</div>}
        </section>
      )}

      {policy && (
        <section className="section">
          <h2 className="section-title">In the viewer</h2>
          <CheckboxRow
            checked={showGhosts}
            disabled={busy}
            onChange={onShowGhosts}
            label={
              <>
                Show policy ghosts {viewerBusy && <Spinner />}
              </>
            }
            helper="Two arms on held-out episodes: the policy in the stock model (grey) and in the calibrated sim (blue). The replay ghost hides meanwhile."
          />
          {busy && <div className="section-hint">Ghosts pause while a policy job runs.</div>}
          {showGhosts && episode != null && heldOut.length > 0 && !heldOut.includes(episode) && (
            <InlineNotice
              tone="info"
              action={
                <Button size="sm" onClick={() => onEpisode(heldOut[0])}>
                  Go to held-out episode {heldOut[0]}
                </Button>
              }
            >
              Episode {episode} is in the policy's training set, so it has no policy ghosts.
            </InlineNotice>
          )}
          {showGhosts && viewerNote && (heldOut.includes(episode ?? -1) || !heldOut.length) && <div className="section-hint">{viewerNote}</div>}
          {showGhosts && viewer && viewer.episode === episode && (
            <div className="section-hint">
              <Badge tone="muted">episode {episode}</Badge>{" "}
              {mode === "shadow"
                ? `${deg(viewer.sims["default"]?.sim_vs_real_deg)} stock vs ${deg(viewer.sims["lock"]?.sim_vs_real_deg)} calibrated against the real arm; ${deg(
                    viewer.sims["lock"]?.divergence_deg,
                  )} between the two sims`
                : `grasp point missed by ${cm(viewer.sims["default"]?.grasp_error_cm)} in the stock model, ${cm(viewer.sims["lock"]?.grasp_error_cm)} in the calibrated sim`}
            </div>
          )}
        </section>
      )}
    </>
  );
}
