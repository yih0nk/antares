import type { EpisodeInfo, ResidualEvent } from "./api";
import { JointTrace, ResidualStrip } from "./Charts";
import type { EpisodeRole } from "./lockInfo";
import type { SimTone } from "./theme";
import { Icon, IconButton, LegendItem, LineKey, Select, SpanKey } from "./ui";

// The bottom strip: a media-player transport (play, speed, previous/next
// episode, the episode picker, time), the selected joint's trace as the
// scrubber, and the residual strip under it on the same time axis.

export const SPEEDS = [0.25, 0.5, 1, 2];

type Props = {
  playing: boolean;
  onPlayPause: () => void;
  speed: number;
  onSpeed: (s: number) => void;
  episodes: EpisodeInfo[];
  episode: number | null;
  onEpisode: (ep: number) => void;
  roleOf: (ep: number) => EpisodeRole | null;
  frame: number;
  frames: number;
  fps: number;
  joint: string;
  jointNames: string[];
  onJoint: (j: string) => void;
  traceReal: number[];
  traceSim: number[];
  simTone: SimTone;
  simLabel: string;
  residualSum: number[];
  events: ResidualEvent[];
  grasps: number[][];
  hasData: boolean;
  onSeek: (f: number) => void;
};

export function episodeOptionLabel(e: EpisodeInfo, role: EpisodeRole | null) {
  return `episode ${e.index} · ${e.duration_s.toFixed(1)}s${role ? ` · ${role}` : ""}`;
}

export function Timeline(p: Props) {
  const i = p.episodes.findIndex((e) => e.index === p.episode);
  const prev = i > 0 ? p.episodes[i - 1] : null;
  const next = i >= 0 && i < p.episodes.length - 1 ? p.episodes[i + 1] : null;
  const secs = (f: number) => (f / Math.max(1, p.fps)).toFixed(2);
  return (
    <footer className="timeline" aria-label="Timeline">
      <div className="transport">
        <IconButton
          label={p.playing ? "Pause" : "Play"}
          shortcut="Space"
          variant="secondary"
          onClick={p.onPlayPause}
          disabled={!p.frames}
        >
          {p.playing ? <Icon.pause /> : <Icon.play />}
        </IconButton>
        <Select label="Speed" hideLabel value={p.speed} onChange={(e) => p.onSpeed(Number(e.target.value))} className="speed-select">
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </Select>
        <span className="transport-sep" aria-hidden="true" />
        <IconButton label={prev ? `Previous episode (${prev.index})` : "Previous episode"} shortcut="[" onClick={() => prev && p.onEpisode(prev.index)} disabled={!prev}>
          <Icon.prev />
        </IconButton>
        <Select
          label="Episode"
          hideLabel
          value={p.episode ?? ""}
          onChange={(e) => p.onEpisode(Number(e.target.value))}
          disabled={!p.episodes.length}
          className="episode-select"
        >
          {p.episodes.map((e) => (
            <option key={e.index} value={e.index}>
              {episodeOptionLabel(e, p.roleOf(e.index))}
            </option>
          ))}
        </Select>
        <IconButton label={next ? `Next episode (${next.index})` : "Next episode"} shortcut="]" onClick={() => next && p.onEpisode(next.index)} disabled={!next}>
          <Icon.next />
        </IconButton>
        <span className="time-readout" aria-label="Time">
          {secs(p.frame)}s <span className="muted">/ {secs(p.frames)}s</span>
        </span>
      </div>

      <div className="strip-head">
        <Select label="Joint" hideLabel value={p.joint} onChange={(e) => p.onJoint(e.target.value)} disabled={!p.jointNames.length} className="joint-select">
          {p.jointNames.map((j) => (
            <option key={j} value={j}>
              {j}
            </option>
          ))}
        </Select>
        <span className="legend-row">
          <LegendItem k={<LineKey tone="real" />}>real</LegendItem>
          <LegendItem k={<LineKey tone={p.simTone} dashed />}>sim · {p.simLabel}</LegendItem>
        </span>
      </div>
      {p.hasData ? (
        <div className="trace-wrap" title="Click or drag to seek · ←/→ one frame · Shift+←/→ one second">
          <JointTrace real={p.traceReal} sim={p.traceSim} frame={p.frame} simTone={p.simTone} fps={p.fps} label={`Playhead, ${p.joint} trace`} onSeek={p.onSeek} />
        </div>
      ) : (
        <div className="trace trace-empty muted">Waiting for the episode…</div>
      )}

      <div className="strip-head">
        <span className="strip-title">|real − sim|, scored joints</span>
        <span className="legend-row">
          <LegendItem k={<SpanKey kind="contact" />}>contact</LegendItem>
          <LegendItem k={<SpanKey kind="anomaly" />}>anomaly</LegendItem>
          <LegendItem k={<SpanKey kind="grasp" />}>gripper closed</LegendItem>
        </span>
      </div>
      {p.hasData && (
        <ResidualStrip sum={p.residualSum} events={p.events} grasps={p.grasps} frames={p.frames} frame={p.frame} simTone={p.simTone} onSeek={p.onSeek} />
      )}
    </footer>
  );
}
