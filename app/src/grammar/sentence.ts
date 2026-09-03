export type WordEntry = { word: string; gapBeforeMs: number | null };
export type SentenceState = { words: WordEntry[]; sentence: string };

export const PAUSE_COMMA_MS = 600;
export const PAUSE_PERIOD_MS = 1400;
export const MAX_WORDS = 40;

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * General para 92k+LSM: acumula tal cual se enseña, sin inventar.
 * Cada palabra guarda gapBeforeMs (tiempo desde anterior). Puntuación se inserta donde ocurrió la pausa, no solo al final.
 * Mayúscula tras . y al inicio. Permite repetir palabra, limita a 40.
 */
export function buildSentenceFromEntries(entries: WordEntry[]): string {
  if (entries.length === 0) return "";
  let out = "";
  for (let i = 0; i < entries.length; i++) {
    const { word, gapBeforeMs } = entries[i];
    if (i === 0) {
      out += capitalize(word);
    } else {
      // puntuación por gap anterior
      const gap = gapBeforeMs ?? 0;
      if (gap >= PAUSE_PERIOD_MS) out += ".";
      else if (gap >= PAUSE_COMMA_MS) out += ",";
      // mayúscula si anterior fue .
      const prevEndedPeriod = (entries[i - 1]?.gapBeforeMs ?? 0) >= PAUSE_PERIOD_MS || out.endsWith(".");
      // actually check if out ends with .
      const needsCap = out.endsWith(".");
      out += " " + (needsCap ? capitalize(word) : word);
    }
  }
  // si último gap es largo, ya se añadió al inicio de siguiente, no duplicar; para display final, si gap actual largo, muestra ,/. al final también
  return out;
}

// compat: words string[] + lastGap -> entries
export function buildSentence(words: string[], lastGapMs: number | null): string {
  const entries: WordEntry[] = words.map((w, i) => ({ word: w, gapBeforeMs: i === 0 ? null : 0 }));
  if (lastGapMs !== null && entries.length > 0) {
    // simula gap final como si hubiera siguiente palabra
    const last = entries[entries.length - 1];
    if (lastGapMs >= PAUSE_PERIOD_MS) return buildSentenceFromEntries(entries) + ".";
    if (lastGapMs >= PAUSE_COMMA_MS) return buildSentenceFromEntries(entries) + ",";
  }
  return buildSentenceFromEntries(entries);
}

export function truncateEntries(entries: WordEntry[], max = MAX_WORDS): WordEntry[] {
  return entries.length > max ? entries.slice(entries.length - max) : entries;
}
