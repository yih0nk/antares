import { useEffect, useState } from "react";
import { api, type FactorInfo, type FactorRegistry } from "./api";
import { CheckboxRow, Disclosure } from "./ui";

// What the fit accounts for: one checkbox per shared-stage factor (built-ins
// on by default, added ones off), how well the data pins each down, and the
// effects antares knows about but does not model yet. Renders nothing when
// the server has no /api/factors route.

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
  onChange: (names: string[] | null) => void;
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
  const other = reg.factors.filter((f) => f.stages.length > 0 && !f.stages.includes("shared"));
  return (
    <Disclosure className="factors" summary="What the fit accounts for" meta={`${on.length} of ${shared.length}`}>
      {shared.map((f) => (
        <CheckboxRow
          key={f.name}
          checked={on.includes(f.name)}
          disabled={disabled}
          onChange={() => toggle(f.name)}
          label={f.label}
          aside={<span className={`pin pin-${f.identifiable}`}>{PINNED[f.identifiable]}</span>}
          helper={f.explains}
          title={`needs: ${f.needs}\n${f.evidence}`}
        />
      ))}
      {selected != null && (
        <button type="button" className="link-btn" onClick={() => onChange(null)} disabled={disabled}>
          Reset to the built-in factors
        </button>
      )}
      {other.length > 0 && (
        <p className="section-hint">
          Also fitted when fitting each joint separately or calibrating the sensor: {other.map((f) => f.label.toLowerCase()).join(", ")}.
        </p>
      )}
      <p className="section-hint">
        Not modelled yet: {reg.not_modelled.map((u) => u.label.toLowerCase()).join(", ")}. Add your own factor with a TOML file (see{" "}
        <code>antares factors</code>).
      </p>
    </Disclosure>
  );
}
