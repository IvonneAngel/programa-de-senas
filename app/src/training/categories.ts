import { DatasetItem } from "../types";

export const DEFAULT_LETRAS: DatasetItem[] = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "Ñ", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
].map((letra) => ({
  id: `letra_${letra}`,
  name: letra,
  category: "letras_y_numeros",
  subType: "letra",
}));

export const DEFAULT_NUMEROS: DatasetItem[] = [
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
].map((num) => ({
  id: `numero_${num}`,
  name: num,
  category: "letras_y_numeros",
  subType: "numero",
}));

export const DEFAULT_PALABRAS: DatasetItem[] = [
  "HOLA",
  "GRACIAS",
  "POR_FAVOR",
  "SI",
  "NO",
  "AYUDA",
  "BIEN",
  "MAL",
  "AMOR",
  "CASA",
  "AGUA",
  "COMIDA"
].map((palabra) => ({
  id: `palabra_${palabra}`,
  name: palabra,
  category: "palabras",
  subType: "palabra",
}));

const CUSTOM_WORDS_KEY = "slp_custom_words_v1";

export function getCustomWords(): DatasetItem[] {
  try {
    const raw = localStorage.getItem(CUSTOM_WORDS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveCustomWord(rawName: string): DatasetItem | null {
  const cleanName = rawName.trim().toUpperCase().replace(/[^A-Z0-9_ÑÁÉÍÓÚ]/g, "_");
  if (!cleanName) return null;

  const current = getCustomWords();
  const id = `palabra_${cleanName}`;
  if (current.some((w) => w.name === cleanName) || DEFAULT_PALABRAS.some((w) => w.name === cleanName)) {
    return null; // Ya existe
  }

  const newItem: DatasetItem = {
    id,
    name: cleanName,
    category: "palabras",
    subType: "palabra",
    custom: true,
  };

  const updated = [...current, newItem];
  localStorage.setItem(CUSTOM_WORDS_KEY, JSON.stringify(updated));
  return newItem;
}

export function deleteCustomWord(id: string): void {
  const current = getCustomWords();
  const updated = current.filter((w) => w.id !== id);
  localStorage.setItem(CUSTOM_WORDS_KEY, JSON.stringify(updated));
}
