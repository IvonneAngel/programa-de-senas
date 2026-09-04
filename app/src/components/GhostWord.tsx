import { useEffect, useState } from "react";
import { buildDicts, parseFrecuencia, rankear, type Dicts } from "../grammar/autocomplete";

type Props = { letra: string | null };

export type Sugerencia = { resto: string; palabra: string };

// Caché única de diccionarios (92k + LSM), rankeado por tecla.
let CACHED: Dicts | null = null;
let loadPromise: Promise<void> | null = null;

async function cargar(): Promise<void> {
  if (CACHED) return;
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    try {
      const [txt, lsmRes, cot, es] = await Promise.all([
        fetch("/diccionario_es_50k.txt").then((r) => (r.ok ? r.text() : "")),
        fetch("/lsm_label_map.json").then((r) => (r.ok ? r.json() : {})),
        fetch("/diccionario_cotidiano.txt").then((r) => (r.ok ? r.text() : "")),
        fetch("/diccionario_lsm_es.txt").then((r) => (r.ok ? r.text() : "")),
      ]);
      const lsm: string[] = [];
      try {
        for (const v of Object.values(lsmRes as Record<string, { word: string }>)) {
          if (v.word.trim()) lsm.push(v.word);
        }
      } catch { /* sin LSM, solo español + 92k */ }
      CACHED = buildDicts(parseFrecuencia(txt.split(/\r?\n/)), lsm, cot.split(/\r?\n/), es.split(/\r?\n/).filter((w) => w.trim() && !w.trim().startsWith("#")));
    } catch {
      CACHED = buildDicts([], [], [], []);
    }
  })();
  return loadPromise;
}

// precarga al importar
cargar();

export default function GhostWord({ letra }: Props) {
  const [sugs, setSugs] = useState<Sugerencia[]>([]);

  useEffect(() => {
    let vivo = true;
    const pref = (letra || "").trim();
    if (!pref) { setSugs([]); return; }
    const aplicar = () => {
      if (!vivo || !CACHED) return;
      const top = rankear(CACHED, pref, 3);
      setSugs(top.map((palabra) => ({ palabra, resto: palabra.slice(pref.length) })));
    };
    if (CACHED) aplicar();
    else cargar().then(aplicar);
    return () => { vivo = false; };
  }, [letra]);

  if (!letra || sugs.length === 0) return null;
  const [mejor, ...alt] = sugs;
  return (
    <span style={{ fontFamily: '"Sheriff Sans", sans-serif' }}>
      <span style={{ opacity: 0.4, color: "gray" }}>{mejor.resto}</span>
      {alt.length > 0 && (
        <span style={{ opacity: 0.25, color: "gray", fontSize: "11px" }}>
          {"  · " + alt.map((a) => a.palabra).join(" · ")}
        </span>
      )}
    </span>
  );
}
