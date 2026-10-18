import { useEffect, useState } from "react";
import { api, type FactorInfo, type FactorRegistry } from "./api";

// What the fit accounts for: one checkbox per shared-stage factor (built-ins
// on by default, added ones off), how well the data pins each down, and the
// effects antares knows about but does not model yet.

const PINNED: Record<FactorInfo["identifiable"], string> = {
  yes: "pinned by the data",
  partly: "partly pinned",
  no: "trades off with others",
  untested: "untested",
};

export const defaultFactors = (reg: FactorRegistry) =>
  reg.factors.filter((f) => f.stages.includes("shared") && f.origin === "builtin").map((f) => f.name);

type Props = {
  selected: string[] | null;
  onChange: (names: string[]) => void;
  disabled?: boolean;
};

export function Factors({ selected, onChange, disabled = false }: Props) {
  const [reg, setReg] = useState<FactorRegistry | null>(null);
  useEffect(() => {
    api.factors().then(setReg).catch(() => setReg(null));
  }, []);
  if (!reg) return null;
  const on = selected ?? defaultFactors(reg);
  const shared = reg.factors.filter((f) => f.stages.includes("shared"));
  const toggle = (name: string) =>
    onChange(on.includes(name) ? on.filter((n) => n !== name) : shared.map((f) => f.name).filter((n) => n === name || on.includes(n)));
  return (
    <details className="factors">
      <summary>
        What the fit accounts for <span className="muted small">{on.length} of {shared.length}</span>
      </summary>
      <ul>
        {shared.map((f) => (
          <li key={f.name} title={`${f.explains}\nneeds: ${f.needs}\n${f.evidence}`}>
            <label className={`toggle ${disabled ? "locked" : ""}`}>
              <input type="checkbox" checked={on.includes(f.name)} disabled={disabled} onChange={() => toggle(f.name)} />
              {f.label}
              <span className={`muted small pin-${f.identifiable}`}>{PINNED[f.identifiable]}</span>
            </label>
          </li>
        ))}
      </ul>
      <p className="muted small">Also fitted with per-joint refinement or sensor calibration:{" "}
        {reg.factors.filter((f) => f.stages.length > 0 && !f.stages.includes("shared")).map((f) => f.label.toLowerCase()).join(", ")}.
      </p>
      <p className="muted small">Not modelled yet: {reg.not_modelled.map((u) => u.label.toLowerCase()).join(", ")}.
        Add your own factor with a TOML file (see <code>antares factors</code>).
      </p>
    </details>
  );
}
