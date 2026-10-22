// Number formatting shared by the viewer's panels.

export const deg = (v: number | null | undefined, digits = 2) => (v == null || !isFinite(v) ? "--" : `${v.toFixed(digits)}°`);
export const cm = (v: number | null | undefined, digits = 1) => (v == null || !isFinite(v) ? "--" : `${v.toFixed(digits)} cm`);
export const short = (id: string) => id.split("/").pop() ?? id;
export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
