# Instrument: the Antares viewer's design system

Chrome is grey; colour means data or status. One primary (filled) button per view.

The viewer is a measuring instrument. Its job is to make one number (the sim-to-real gap) and the things that move it legible, so the interface stays quiet and every colour on screen tells you something about the robot or the sim.

## Where to change things

| What | Where | Notes |
| --- | --- | --- |
| Colours | `src/theme.ts` (`palette`) | The single source of truth. `applyTheme()` (called in `src/main.tsx` before the first render) writes each key onto `:root` as a CSS custom property (`borderStrong` becomes `--border-strong`). The three.js materials (`src/Arm.tsx`, `src/Scene.tsx`), the 3D background and grid (`src/Viewport.tsx`) and the cross-robot heatmap (`src/Charts.tsx`) import the hex values directly. `src/styles.css` contains no colour literals; it only uses the custom properties. |
| Type, spacing, radii, motion, layout sizes | `:root` in `src/styles.css` | Not shared with three.js, so CSS owns them. |
| Components | `src/ui.tsx` | One implementation of each control (list below). Style classes live in `styles.css` under the same names. |
| Font | `src/main.tsx` imports `@fontsource/ibm-plex-mono` 400, 500 and 600 | Self-hosted; no Google Fonts request. |

After changing a palette value, re-check the contrast table below. `index.html` sets `<meta name="color-scheme" content="dark">` so the browser paints a dark canvas before the theme is applied.

## Colour

Dark only, single theme.

### Surfaces and text

| Token | Hex | Use |
| --- | --- | --- |
| `--bg` | `#0E1013` | Page and 3D viewport background (also the three.js clear colour), chart wells, run cards |
| `--surface` | `#15181D` | Topbar, inspector, timeline, tiles |
| `--raised` | `#1C2026` | Controls, selected rows, popovers, menus, modals, toasts |
| `--border` | `#2B3039` | Dividers, quiet outlines |
| `--border-strong` | `#3B414C` | Control outlines, selected-row outline, the gripper-closed hatch |
| `--text` | `#ECEEF2` | Body text, the focus ring |
| `--text-muted` | `#A1A8B5` | Secondary text, captions, unselected controls |
| `--text-disabled` | `#737B88` | Disabled controls only. Never essential information. |
| `--accent` / `--on-accent` | `#ECEEF2` / `#0E1013` | The primary button's fill and label |

Measured contrast (WCAG 2.x ratio):

| Text | on `--bg` | on `--surface` | on `--raised` |
| --- | --- | --- | --- |
| `--text` | 16.4 | 15.3 | 14.1 |
| `--text-muted` | 8.0 | 7.4 | 6.8 |
| `--text-disabled` | 4.5 | 4.2 | 3.8 |
| `--on-accent` on `--accent` | 16.4 | | |

### Data and status

| Token | Hex | Means | On `--bg` |
| --- | --- | --- | --- |
| `--real` | `#E6E8EC` | The logged real arm | 15.5 |
| `--stock` | `#7C8494` | The sim on the stock model (uncalibrated MJCF defaults) | 5.1 |
| `--sim` | `#4D8DFF` | The sim on calibrated physics: a saved lock or the live fit | 6.0 |
| `--comp` | `#C86BC4` | The sim driven by lag-compensated commands | 5.8 |
| `--good` | `#43C38A` | An improvement (the gap shrank, a check passed) | 8.5 |
| `--warn` | `#E8B341` | An anomaly, a partial run, a warning | 9.9 |
| `--critical` | `#FF6A55` | Contact, the sim disagrees, an error, a drifted check, a destructive action | 6.8 |

Every data and status colour is at least 3:1 on every surface. `--sim` and `--comp` stay separable under protanopia and deuteranopia (ΔE 8.8) as well as for normal vision (ΔE 19.3).

## Encoding rules (never colour alone)

- **Real arm:** solid and opaque, `--real`. **Sim arms:** translucent. In 2D traces the real arm is a solid line and a sim a dashed line; legends draw the line style (`LineKey`) or the opacity (`ArmKey`), not just a swatch.
- **A sim's colour follows its physics.** Stock model: `--stock`. Any calibrated lock or the live fit: `--sim`. This applies to the 3D ghost, the joint trace, the residual area, the swatch on each Sim physics row, and the per-joint bars (grey bar: stock-model error; blue bar: the selected physics' error, with a legend).
- **Policy ghosts:** the policy in the stock model is `--stock`, in the calibrated sim `--sim`. While they are drawn the replay ghost hides (two blue arms would be ambiguous) and the legend says so.
- **Lag-compensated arm:** `--comp`.
- **Where the sim disagrees:** contact spans and the ghost's glow are `--critical`; anomalies are `--warn`; gripper-closed windows are a neutral `--border-strong` hatch. The glowing ghost also tints its base colour toward the status colour so a glowing blue arm never reads as the magenta lag-compensated one.
- **Status always has a glyph and a word:** the status pill (icon + "Sim disagrees: contact on …"), badges (`◐ partial · 70/150`, `✓ ok`, `▲ drifted`), the gap delta (`▼ from 4.44° on the stock model`), notice icons.
- **Selected is never colour alone:** a check glyph, weight 500, a border or a filled radio marks it (segmented options, tabs, the Sim physics radio rows, the selected joint's inset bar, the pressed Compare episode button).

## Typography

IBM Plex Mono for everything, including Fleet and anything drawn over the 3D view. No sans-serif anywhere. Fallback: `ui-monospace, "SF Mono", Menlo, monospace`.

| Size | Use |
| --- | --- |
| 11px | Uppercase section labels (letter-spacing 0.06em), badges, small captions |
| 12px | Secondary text, captions, helper lines, legends, tables |
| 13px | Body default, buttons, inputs |
| 15px | Section titles |
| 36px, weight 600 | The hero gap number only |

Weights: 400 body, 500 buttons and emphasis, 600 the hero only. Mono is wide, so keep labels short. Numbers use tabular figures (`.num`).

## Space, shape, motion

- 4px grid: 4, 8, 12, 16, 24, 32 (`--s1` to `--s6`).
- Radius 4 (chips, badges, bars), 6 (controls), 8 (popovers, menus, modal, cards).
- 1px borders. No shadows except popovers, menus, toasts and the modal.
- Control height 28px; hit targets at least 24px.
- Motion: 120ms ease-out for hover and press. `prefers-reduced-motion` turns off the ghost's tween onto a new fit, eases the glow instantly, stops spinners spinning (they become a dotted ring) and freezes the indeterminate progress bar.
- Focus ring: 2px `--text`, 2px offset, on `:focus-visible`.

## Components (`src/ui.tsx`)

| Component | Notes |
| --- | --- |
| `Button` | `primary` / `secondary` / `ghost` / `danger`; `busy` shows a spinner and disables it, and the caller passes the "-ing" label ("Calibrating…"). One primary per view: the action the user most likely wants next. |
| `ButtonLink` | A download that looks like a button. |
| `IconButton` | Square, `label` is the accessible name and tooltip, `shortcut` adds "(Space)" to the tooltip and `aria-keyshortcuts`. |
| `SegmentedControl` | Mutually exclusive options as pressed buttons (`aria-pressed`). |
| `Select` | Native select with a chevron and a (possibly visually hidden) label. |
| `CheckboxRow` | Native checkbox, a label, an optional aside ("+600 evals") and a one-line helper wired to `aria-describedby`. |
| `Disclosure` | Styled `details`/`summary` with a caret and a meta line ("defaults", "2 changed"). |
| `Tabs` / `TabPanel` | WAI-ARIA tabs, automatic activation, arrow keys and Home/End. Inactive panels stay mounted (hidden) so jobs keep streaming. |
| `Stat` | Label, hero number, delta line, caption. |
| `Badge` | Neutral, `muted`, `good`, `warn`, `critical`; optional glyph. Used for held-out/train, partial, kind (shared / per-joint / sensor), repeat, Fleet verdicts. |
| `InlineNotice` | `info` / `warn` / `error`: what happened and how to fix it, with the API's message kept. Dismissible. |
| `ProgressBar` | Determinate or indeterminate, with `role=progressbar`. |
| `Spinner` | Inline busy mark. |
| `EmptyState` | Why it's empty and the action that fills it. |
| `useToasts` / `ToastHost` | Toasts with an optional action (Undo). Paused while hovered or focused; closing one counts as letting it expire. |
| `OverflowMenu` | Row actions (`role=menu`): arrow keys, Escape returns focus to the button. |
| `LineKey`, `ArmKey`, `SpanKey`, `LegendItem` | Legend marks that show line style and opacity, not just colour. |

Feature components built on these: `GapStat`, `PhysicsPicker`, `JointBars`, `CalibrateSection`, `ParamsDisclosure` (`src/CalibrateTab.tsx`), `CameraView`, `PolicyTab`, `ToolsTab`, `CrossModal`, `Timeline`, `ArmScene` (`src/Viewport.tsx`).

## Layout

```
┌ topbar: antares · Viewer | Fleet ·············· dataset chip · dataset · Add dataset ┐
├ viewport (3D) ──────────────────────────────────────────┬ inspector (360px) ─────────┤
│ status pill (only during an event)   Show: Both|Real|Sim │ Calibrate Camera Policy Tools│
│                                      [Compare episode]   │                            │
│ legend                                                   │                            │
├ timeline: ▶ speed | ⏮ episode ⏭ | time                  │                            │
│ joint trace (scrubber) · residual strip                  │                            │
└──────────────────────────────────────────────────────────┴────────────────────────────┘
```

- **Calibrate** (default tab): the gap (hero number, delta from the stock model, caption naming the episode and whether it is held out or train for the selected physics' split); Sim physics (Stock model, the live fit while calibrating, then locks newest first with kind, held-out error, created time and partial/repeat badges; four shown, then "Show all"; a row menu with Download MJCF and Delete); per-joint bars; the Calibrate button, its run state and result, an Options disclosure; a Parameters disclosure.
- **Camera:** three steps (camera, scene, objects), each saved per lock. A step that needs an earlier one says "Needs: camera fit" instead of showing a disabled button. A lock without a camera fit shows an empty state that can start a fit or switch to a lock that has one.
- **Policy:** train, evaluate (Shadow / Closed loop, each explained in one line), results, policy ghosts.
- **Tools:** lag compensation, the cross-robot matrix, export.
- Narrow screens (≤ 900px) stack topbar, viewport, timeline and inspector, and the page scrolls.

## Keyboard

| Key | Action |
| --- | --- |
| Space | Play / pause (ignored in inputs and selects; a keyboard-focused button keeps Space for itself) |
| ← / → | Step one frame (pauses) |
| Shift + ← / → | Step one second |
| [ / ] | Previous / next episode |

Buttons mention their shortcut in the tooltip. Tabs use arrow keys while focused.

## Error prevention and recovery

- **Delete is soft.** The API's `DELETE /api/locks/{name}` unlinks the file at once and has no restore, so the row hides and a toast offers Undo for 6 seconds; the request goes out only when the toast expires (or is closed). Closing the tab inside the window keeps the lock. If the deleted lock was selected the ghost falls back to the Stock model, and Undo re-selects it.
- **No unexplained disabled control.** A disabled control says why next to it or in its tooltip ("Dataset locked while calibrating", "Waiting for the episode to load", "Needs: scene fit").
- **Errors** are inline notices that keep the API's message and say what to do next. They can be dismissed.
- Switching datasets or physics mid-calibration stays blocked, with the reason shown.

## Copy

Sentence case, verbs first on buttons, plain words. Current names: "Sim physics" (not "Ghost parameters"), "Stock model" (not "Default"), "Compare episode" (not "Side by side"), "Both / Real / Sim", "Calibrate", "Fit each joint separately", "Fit gravity sag", "Fit servo deadband", "Held-out (not used in the fit)", "Stop and save best so far".
