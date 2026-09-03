import { useEffect, useState } from "react";

type Props = { letra: string | null };

// ponytail: diccionario grande 92k + LSM 249, sin demo 7. Carga vía fetch desde /diccionario_grande.txt y /lsm_label_map.json
let DICCIONARIO_GRANDE: string[] | null = null;
let LSM_PALABRAS: string[] = [];

async function cargarDiccionarios(): Promise<string[]> {
  if (DICCIONARIO_GRANDE) return [...DICCIONARIO_GRANDE, ...LSM_PALABRAS];
  try {
    const [txt, lsmRes] = await Promise.all([
      fetch("/diccionario_grande.txt").then((r) => (r.ok ? r.text() : "")),
      fetch("/lsm_label_map.json").then((r) => (r.ok ? r.json() : {})),
    ]);
    const palabras = txt
      .split(/\r?\n/)
      .map((w) => w.trim().toLowerCase())
      .filter((w) => w.length >= 2);
    // LSM 249 del label_map
    try {
      const lsm = Object.values(lsmRes as Record<string, { word: string }>).map((v) => v.word.toLowerCase().trim()).filter(Boolean);
      LSM_PALABRAS = [...new Set(lsm)];
    } catch {
      LSM_PALABRAS = [];
    }
    DICCIONARIO_GRANDE = [...new Set([...palabras, ...LSM_PALABRAS])];
    return [...DICCIONARIO_GRANDE, ...LSM_PALABRAS];
  } catch {
    DICCIONARIO_GRANDE = [];
    return [];
  }
}

// precarga al importar
cargarDiccionarios();

export default function GhostWord({ letra }: Props) {
  const [sugerencia, setSugerencia] = useState<string | null>(null);

  useEffect(() => {
    if (!letra) { setSugerencia(null); return; }
    // letra puede ser prefijo de varias letras: "H" -> "ola", "HO" -> "la" (hola) o "spital" (hospital)
    const lower = letra.toLowerCase().trim();
    if (!lower) { setSugerencia(null); return; }
    const buscar = (dict: string[]) => {
      const matches = dict.filter((w) => w.startsWith(lower)).sort((a, b) => b.length - a.length).slice(0, 3);
      const match = matches[0];
      return match ? match.slice(letra.length) : null;
    };
    if (DICCIONARIO_GRANDE) {
      setSugerencia(buscar([...DICCIONARIO_GRANDE, ...LSM_PALABRAS]));
    } else {
      cargarDiccionarios().then((dict) => setSugerencia(buscar(dict)));
    }
  }, [letra]);

  if (!letra || !sugerencia) return null;
  return (
    <span style={{ opacity: 0.4, color: "gray", fontFamily: '"Sheriff Sans", sans-serif' }}>
      {sugerencia} {/* fantasma Sheriff Sans */}
    </span>
  );
}
