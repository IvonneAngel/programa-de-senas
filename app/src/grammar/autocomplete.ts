/** Lógica pura del autocompletado (sin DOM/fetch): testeable en node. */

export function norm(s: string): string {
  return s.toLowerCase().trim().normalize("NFD").replace(/[̀-ͯ]/g, "");
}

export type Dicts = { dict: string[]; dictSet: Set<string>; tier: Map<string, number>; freq: Map<string, number> };

const clean = (ws: string[]) => ws.map((w) => w.trim().toLowerCase()).filter((w) => w.length >= 2);

/** Lineas "palabra frecuencia" del es_50k -> solo palabras, en orden de frecuencia. */
export function parseFrecuencia(lineas: string[]): string[] {
  const out: string[] = [];
  for (const ln of lineas) {
    const w = ln.split(/\s+/)[0]?.trim().toLowerCase() ?? "";
    if (w.length >= 2) out.push(w);
  }
  return [...new Set(out)];
}

/** Tiers: 0 LSM + español curado (dominio) > 1 español frecuencia. Sin inglés. */
export function buildDicts(es50k: string[], lsm: string[], cotidiano: string[] = [], lsmEs: string[] = []): Dicts {
  const lsmClean = [...new Set([...clean(lsm), ...clean(lsmEs)])];
  const cotClean = [...new Set(clean(cotidiano))];
  const base = [...new Set(clean(es50k))];
  const dict = [...new Set([...lsmClean, ...cotClean, ...base])];
  const tier = new Map<string, number>();
  const freq = new Map<string, number>();
  base.forEach((w, i) => { tier.set(w, 1); freq.set(w, i); });
  for (const w of cotClean) if (!tier.has(w)) tier.set(w, 1);
  for (const w of lsmClean) tier.set(w, 0);
  return { dict, dictSet: new Set(dict), tier, freq };
}

/**
 * Rank: 1) tier (dominio > español), 2) frecuencia (la que más se usa), 3) más corta, 4) alfabético.
 * Casos: prefijo vacío -> []; sin coincidencias -> [].
 * OJO: aunque el prefijo ya sea palabra ("ho","si","no"), SE SIGUE sugiriendo:
 * el ghost nunca bloquea (si terminó, ignora y pausa; si sigue, ayuda). Callar era peor.
 */
export function rankear(d: Dicts, prefijo: string, limite = 3): string[] {
  const p = norm(prefijo);
  if (!p) return [];
  const cands: string[] = [];
  for (const w of d.dict) {
    if (w !== p && norm(w).startsWith(p)) cands.push(w);
    if (cands.length >= 600) break;
  }
  const t = (w: string) => d.tier.get(w) ?? 1;
  const f = (w: string) => d.freq.get(w) ?? Number.MAX_SAFE_INTEGER;
  cands.sort((a, b) => {
    if (t(a) !== t(b)) return t(a) - t(b);
    if (f(a) !== f(b)) return f(a) - f(b);
    if (a.length !== b.length) return a.length - b.length;
    return a < b ? -1 : 1;
  });
  return cands.slice(0, limite);
}
