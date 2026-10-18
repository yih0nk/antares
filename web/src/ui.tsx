// The Instrument component set: one implementation of each control, used by
// every view. Styling lives in styles.css under the matching class names;
// colours come from theme.ts through CSS custom properties. See web/DESIGN.md.

import {
  useEffect,
  useId,
  useRef,
  useState,
  type AnchorHTMLAttributes,
  type ButtonHTMLAttributes,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";

// ---- icons (inline SVG so they render in any font) ---------------------------

function Svg({ children, size = 14 }: { children: ReactNode; size?: number }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      {children}
    </svg>
  );
}

export const Icon = {
  play: () => (
    <Svg>
      <path d="M4.5 2.5v11l9-5.5z" fill="currentColor" />
    </Svg>
  ),
  pause: () => (
    <Svg>
      <rect x="3.5" y="2.5" width="3" height="11" rx="0.5" fill="currentColor" />
      <rect x="9.5" y="2.5" width="3" height="11" rx="0.5" fill="currentColor" />
    </Svg>
  ),
  prev: () => (
    <Svg>
      <rect x="3" y="3" width="1.75" height="10" fill="currentColor" />
      <path d="M13 3v10L5.5 8z" fill="currentColor" />
    </Svg>
  ),
  next: () => (
    <Svg>
      <rect x="11.25" y="3" width="1.75" height="10" fill="currentColor" />
      <path d="M3 3v10L10.5 8z" fill="currentColor" />
    </Svg>
  ),
  more: () => (
    <Svg>
      <circle cx="3.5" cy="8" r="1.3" fill="currentColor" />
      <circle cx="8" cy="8" r="1.3" fill="currentColor" />
      <circle cx="12.5" cy="8" r="1.3" fill="currentColor" />
    </Svg>
  ),
  close: () => (
    <Svg>
      <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </Svg>
  ),
  check: () => (
    <Svg>
      <path d="M3 8.5l3.2 3L13 4.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  ),
  chevron: () => (
    <Svg size={12}>
      <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  ),
  info: () => (
    <Svg>
      <circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 7v4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="8" cy="4.75" r="1" fill="currentColor" />
    </Svg>
  ),
  warn: () => (
    <Svg>
      <path d="M8 2L14.5 13.5h-13z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M8 6.5v3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="8" cy="11.75" r="0.9" fill="currentColor" />
    </Svg>
  ),
  error: () => (
    <Svg>
      <circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M5.75 5.75l4.5 4.5M10.25 5.75l-4.5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </Svg>
  ),
  download: () => (
    <Svg>
      <path d="M8 2.5v8M4.5 7l3.5 3.5L11.5 7M3 13.5h10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  ),
};

// ---- busy spinner ---------------------------------------------------------------

export function Spinner({ label }: { label?: string }) {
  return label ? <span className="spinner" role="status" aria-label={label} /> : <span className="spinner" aria-hidden="true" />;
}

// ---- buttons --------------------------------------------------------------------

type Variant = "primary" | "secondary" | "ghost" | "danger";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: "sm" | "md";
  // Shows a spinner in place of the icon and disables the button; pass the
  // "-ing" label as children while busy.
  busy?: boolean;
  icon?: ReactNode;
  block?: boolean;
};

const btnClass = (variant: Variant, size: "sm" | "md", block?: boolean, extra = "") =>
  ["btn", `btn-${variant}`, size === "sm" ? "btn-sm" : "", block ? "btn-block" : "", extra].filter(Boolean).join(" ");

export function Button({ variant = "secondary", size = "md", busy = false, icon, block, className, children, disabled, ...rest }: ButtonProps) {
  return (
    <button type="button" className={btnClass(variant, size, block, className)} disabled={disabled || busy} aria-busy={busy || undefined} {...rest}>
      {busy ? <Spinner /> : icon}
      {children != null && children !== false && <span className="btn-label">{children}</span>}
    </button>
  );
}

// A link that looks like a button (downloads).
export function ButtonLink({
  variant = "secondary",
  size = "md",
  icon,
  className,
  children,
  ...rest
}: AnchorHTMLAttributes<HTMLAnchorElement> & { variant?: Variant; size?: "sm" | "md"; icon?: ReactNode }) {
  return (
    <a className={btnClass(variant, size, false, className)} {...rest}>
      {icon}
      <span className="btn-label">{children}</span>
    </a>
  );
}

// A square icon-only button; `label` is its accessible name and tooltip,
// `shortcut` is appended to the tooltip and exposed as aria-keyshortcuts.
export function IconButton({
  label,
  shortcut,
  variant = "ghost",
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; shortcut?: string; variant?: Variant }) {
  return (
    <button
      type="button"
      className={btnClass(variant, "md", false, `btn-icon ${className ?? ""}`)}
      aria-label={label}
      title={shortcut ? `${label} (${shortcut})` : label}
      aria-keyshortcuts={shortcut}
      {...rest}
    >
      {children}
    </button>
  );
}

// ---- segmented control ------------------------------------------------------------

export type SegOption<T extends string> = { value: T; label: ReactNode; title?: string; disabled?: boolean };

// Mutually exclusive options as pressed/unpressed buttons. The selected one is
// marked by weight, border and fill, never by colour alone.
export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  label,
  ariaLabel,
  className,
}: {
  options: SegOption<T>[];
  value: T;
  onChange: (v: T) => void;
  label?: ReactNode;
  ariaLabel?: string;
  className?: string;
}) {
  const id = useId();
  return (
    <div className={`seg-wrap ${className ?? ""}`}>
      {label && (
        <span className="seg-label" id={`${id}-label`}>
          {label}
        </span>
      )}
      <div className="segmented" role="group" aria-label={ariaLabel} aria-labelledby={label && !ariaLabel ? `${id}-label` : undefined}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            className={`seg ${o.value === value ? "on" : ""}`}
            aria-pressed={o.value === value}
            disabled={o.disabled}
            title={o.title}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// ---- select -------------------------------------------------------------------------

export function Select({
  label,
  hideLabel = false,
  className,
  children,
  ...rest
}: SelectHTMLAttributes<HTMLSelectElement> & { label?: string; hideLabel?: boolean }) {
  const id = useId();
  return (
    <span className={`select-wrap ${className ?? ""}`}>
      {label && (
        <label htmlFor={id} className={hideLabel ? "sr-only" : "select-label"}>
          {label}
        </label>
      )}
      <span className="select">
        <select id={id} {...rest}>
          {children}
        </select>
        <Icon.chevron />
      </span>
    </span>
  );
}

// ---- checkbox row with a one-line helper ---------------------------------------------

export function CheckboxRow({
  checked,
  onChange,
  label,
  helper,
  aside,
  disabled = false,
  title,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  helper?: ReactNode;
  aside?: ReactNode;
  disabled?: boolean;
  title?: string;
}) {
  const id = useId();
  return (
    <div className={`check-row ${disabled ? "disabled" : ""}`} title={title}>
      <input
        id={id}
        type="checkbox"
        className="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        aria-describedby={helper ? `${id}-help` : undefined}
      />
      <label htmlFor={id} className="check-label">
        <span>{label}</span>
        {aside && <span className="check-aside">{aside}</span>}
      </label>
      {helper && (
        <span id={`${id}-help`} className="check-help">
          {helper}
        </span>
      )}
    </div>
  );
}

// ---- disclosure (details/summary) ------------------------------------------------------

export function Disclosure({
  summary,
  meta,
  defaultOpen = false,
  className,
  children,
}: {
  summary: ReactNode;
  meta?: ReactNode;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details className={`disclosure ${className ?? ""}`} open={defaultOpen || undefined}>
      <summary>
        <span className="disc-caret">
          <Icon.chevron />
        </span>
        <span className="disc-title">{summary}</span>
        {meta != null && <span className="disc-meta">{meta}</span>}
      </summary>
      <div className="disc-body">{children}</div>
    </details>
  );
}

// ---- tabs ---------------------------------------------------------------------------------

export type TabDef<T extends string> = { id: T; label: ReactNode; badge?: ReactNode };

// WAI-ARIA tabs with automatic activation: arrow keys move and select,
// Home/End jump to the ends. Pair with <TabPanel idBase=… id=…>.
export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
  ariaLabel,
  idBase,
  className,
}: {
  tabs: TabDef<T>[];
  value: T;
  onChange: (v: T) => void;
  ariaLabel: string;
  idBase: string;
  className?: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: ReactKeyboardEvent<HTMLButtonElement>, i: number) => {
    const n = tabs.length;
    let j = -1;
    if (e.key === "ArrowRight") j = (i + 1) % n;
    else if (e.key === "ArrowLeft") j = (i - 1 + n) % n;
    else if (e.key === "Home") j = 0;
    else if (e.key === "End") j = n - 1;
    if (j < 0) return;
    e.preventDefault();
    onChange(tabs[j].id);
    refs.current[j]?.focus();
  };
  return (
    <div role="tablist" aria-label={ariaLabel} className={`tabs ${className ?? ""}`}>
      {tabs.map((t, i) => (
        <button
          key={t.id}
          ref={(el) => (refs.current[i] = el)}
          type="button"
          role="tab"
          id={`${idBase}-tab-${t.id}`}
          aria-selected={t.id === value}
          aria-controls={`${idBase}-panel-${t.id}`}
          tabIndex={t.id === value ? 0 : -1}
          className={`tab ${t.id === value ? "on" : ""}`}
          onClick={() => onChange(t.id)}
          onKeyDown={(e) => onKey(e, i)}
        >
          {t.label}
          {t.badge}
        </button>
      ))}
    </div>
  );
}

// A tab's panel. Inactive panels stay mounted (hidden) so running jobs keep
// streaming while another tab is open.
export function TabPanel({
  idBase,
  id,
  active,
  className,
  children,
}: {
  idBase: string;
  id: string;
  active: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div role="tabpanel" id={`${idBase}-panel-${id}`} aria-labelledby={`${idBase}-tab-${id}`} hidden={!active} className={className}>
      {children}
    </div>
  );
}

// ---- stat (label, hero number, delta line, caption) -----------------------------------------

export type Tone = "neutral" | "muted" | "good" | "warn" | "critical" | "sim" | "stock" | "comp";

export function Stat({
  label,
  value,
  delta,
  caption,
  extra,
}: {
  label: ReactNode;
  value: ReactNode;
  delta?: { glyph: string; text: ReactNode; tone: Tone } | null;
  caption?: ReactNode;
  extra?: ReactNode;
}) {
  return (
    <div className="stat">
      <div className="section-label">{label}</div>
      <div className="stat-value">{value}</div>
      {delta && (
        <div className={`stat-delta tone-${delta.tone}`}>
          <span aria-hidden="true">{delta.glyph}</span> {delta.text}
        </div>
      )}
      {caption && <div className="stat-caption">{caption}</div>}
      {extra}
    </div>
  );
}

// ---- badge --------------------------------------------------------------------------------------

export function Badge({ tone = "neutral", glyph, title, children }: { tone?: Tone; glyph?: string; title?: string; children: ReactNode }) {
  return (
    <span className={`badge tone-${tone}`} title={title}>
      {glyph && (
        <span className="badge-glyph" aria-hidden="true">
          {glyph}
        </span>
      )}
      {children}
    </span>
  );
}

// ---- inline notice ------------------------------------------------------------------------------

export function InlineNotice({
  tone = "info",
  title,
  children,
  onDismiss,
  action,
  className,
}: {
  tone?: "info" | "warn" | "error";
  title?: ReactNode;
  children?: ReactNode;
  onDismiss?: () => void;
  action?: ReactNode;
  className?: string;
}) {
  const icon = tone === "error" ? <Icon.error /> : tone === "warn" ? <Icon.warn /> : <Icon.info />;
  return (
    <div className={`notice notice-${tone} ${className ?? ""}`} role={tone === "error" ? "alert" : "status"}>
      <span className="notice-icon">{icon}</span>
      <div className="notice-body">
        {title && <div className="notice-title">{title}</div>}
        {children && <div className="notice-text">{children}</div>}
        {action && <div className="notice-action">{action}</div>}
      </div>
      {onDismiss && (
        <IconButton label="Dismiss" className="notice-close" onClick={onDismiss}>
          <Icon.close />
        </IconButton>
      )}
    </div>
  );
}

// ---- progress bar --------------------------------------------------------------------------------

// `value` in [0, 1]; null draws an indeterminate bar.
export function ProgressBar({ value, label }: { value: number | null; label: string }) {
  const pct = value == null ? null : Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      className={`progress ${pct == null ? "indeterminate" : ""}`}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
    >
      <span style={pct == null ? undefined : { width: `${pct}%` }} />
    </div>
  );
}

// ---- empty state (why + action) -------------------------------------------------------------------

export function EmptyState({ title, children, actions }: { title: ReactNode; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      {children && <div className="empty-body">{children}</div>}
      {actions && <div className="empty-actions">{actions}</div>}
    </div>
  );
}

// ---- toast with optional undo -----------------------------------------------------------------------

export type ToastSpec = {
  message: ReactNode;
  tone?: "info" | "good" | "critical";
  // Milliseconds on screen; paused while hovered or focused.
  duration?: number;
  action?: { label: string; onClick: () => void };
  // Runs when the toast leaves without its action: timed out or closed.
  onExpire?: () => void;
};

export type ToastItem = ToastSpec & { id: number };

export function useToasts() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const next = useRef(1);
  const push = (t: ToastSpec) => {
    const id = next.current++;
    setToasts((ts) => [...ts, { ...t, id }]);
    return id;
  };
  const remove = (id: number) => setToasts((ts) => ts.filter((t) => t.id !== id));
  return { toasts, push, remove };
}

function Toast({ t, onRemove }: { t: ToastItem; onRemove: (id: number) => void }) {
  const [paused, setPaused] = useState(false);
  const remaining = useRef(t.duration ?? 5000);
  const done = useRef(false);
  const finish = (expire: boolean) => {
    if (done.current) return;
    done.current = true;
    if (expire) t.onExpire?.();
    onRemove(t.id);
  };
  useEffect(() => {
    if (paused) return;
    const started = Date.now();
    const id = window.setTimeout(() => finish(true), Math.max(0, remaining.current));
    return () => {
      window.clearTimeout(id);
      remaining.current -= Date.now() - started;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paused]);
  const icon = t.tone === "critical" ? <Icon.error /> : t.tone === "good" ? <Icon.check /> : <Icon.info />;
  return (
    <div
      className={`toast tone-${t.tone ?? "info"}`}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <span className="toast-icon">{icon}</span>
      <span className="toast-text">{t.message}</span>
      {t.action && (
        <Button
          size="sm"
          variant="secondary"
          onClick={() => {
            t.action!.onClick();
            finish(false);
          }}
        >
          {t.action.label}
        </Button>
      )}
      <IconButton label="Close" className="btn-sm" onClick={() => finish(true)}>
        <Icon.close />
      </IconButton>
    </div>
  );
}

export function ToastHost({ toasts, onRemove }: { toasts: ToastItem[]; onRemove: (id: number) => void }) {
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((t) => (
        <Toast key={t.id} t={t} onRemove={onRemove} />
      ))}
    </div>
  );
}

// ---- overflow menu --------------------------------------------------------------------------------

export type MenuItem = {
  label: string;
  onSelect?: () => void;
  href?: string;
  download?: string;
  danger?: boolean;
  icon?: ReactNode;
};

export function OverflowMenu({ label, header, items, disabled }: { label: string; header?: ReactNode; items: MenuItem[]; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement | null>(null);
  const menu = useRef<HTMLDivElement | null>(null);
  const id = useId();

  useEffect(() => {
    if (!open) return;
    menu.current?.querySelector<HTMLElement>("[role=menuitem]")?.focus();
    const onDown = (e: MouseEvent) => {
      if (!menu.current?.contains(e.target as Node) && !btn.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [open]);

  const close = (refocus: boolean) => {
    setOpen(false);
    if (refocus) btn.current?.focus();
  };

  const onKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const els = Array.from(menu.current?.querySelectorAll<HTMLElement>("[role=menuitem]") ?? []);
    const i = els.indexOf(document.activeElement as HTMLElement);
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      close(true);
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const j = e.key === "ArrowDown" ? (i + 1) % els.length : (i - 1 + els.length) % els.length;
      els[j]?.focus();
    } else if (e.key === "Tab") {
      setOpen(false);
    }
  };

  return (
    <div className="menu-wrap">
      <button
        ref={btn}
        type="button"
        className="btn btn-ghost btn-icon btn-sm"
        aria-label={label}
        title={label}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
      >
        <Icon.more />
      </button>
      {open && (
        <div className="menu" role="menu" id={id} ref={menu} onKeyDown={onKey}>
          {header && <div className="menu-header">{header}</div>}
          {items.map((it) =>
            it.href ? (
              <a
                key={it.label}
                role="menuitem"
                className={`menu-item ${it.danger ? "danger" : ""}`}
                href={it.href}
                download={it.download}
                onClick={() => close(false)}
              >
                {it.icon}
                {it.label}
              </a>
            ) : (
              <button
                key={it.label}
                type="button"
                role="menuitem"
                className={`menu-item ${it.danger ? "danger" : ""}`}
                onClick={() => {
                  close(true);
                  it.onSelect?.();
                }}
              >
                {it.icon}
                {it.label}
              </button>
            ),
          )}
        </div>
      )}
    </div>
  );
}

// ---- legend keys: line style, arm opacity, span fills --------------------------------------------

// A trace's line in a legend: solid for the real arm, dashed for a sim.
export function LineKey({ tone, dashed = false }: { tone: "real" | "stock" | "sim" | "comp" | "critical"; dashed?: boolean }) {
  return (
    <svg className="key line-key" width="20" height="8" viewBox="0 0 20 8" aria-hidden="true" focusable="false">
      <line x1="1" x2="19" y1="4" y2="4" style={{ stroke: `var(--${tone})` }} strokeWidth="2" strokeDasharray={dashed ? "4 3" : undefined} />
    </svg>
  );
}

// An arm in the 3D legend: solid (the real arm) or translucent with an outline (a sim).
export function ArmKey({ tone, translucent = false }: { tone: "real" | "stock" | "sim" | "comp"; translucent?: boolean }) {
  return <span className={`key arm-key ${translucent ? "translucent" : ""}`} style={{ ["--key" as string]: `var(--${tone})` }} aria-hidden="true" />;
}

// A span kind in the residual strip's legend.
export function SpanKey({ kind }: { kind: "contact" | "anomaly" | "grasp" }) {
  return <span className={`key span-key ${kind}`} aria-hidden="true" />;
}

export function LegendItem({ k, children, title }: { k: ReactNode; children: ReactNode; title?: string }) {
  return (
    <span className="legend-item" title={title}>
      {k}
      <span>{children}</span>
    </span>
  );
}
