import JSZip from "jszip";
import { DatasetCategory } from "../types";

const DB_NAME = "SignLanguageDatasetDB";
const STORE_NAME = "frames";
const DB_VERSION = 1;

interface StoredFrame {
  id: string; // e.g. "letras_y_numeros/letra_A/1700000000000_1"
  category: DatasetCategory;
  itemId: string;
  itemName: string;
  dataUrl: string;
  timestamp: number;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("category", "category", { unique: false });
        store.createIndex("itemId", "itemId", { unique: false });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Guarda una ráfaga de frames capturados para una categoría e ítem específico
 */
export async function saveItemFrames(
  category: DatasetCategory,
  itemId: string,
  itemName: string,
  framesDataUrl: string[]
): Promise<number> {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);

  const now = Date.now();
  for (let i = 0; i < framesDataUrl.length; i++) {
    const item: StoredFrame = {
      id: `${category}__${itemId}__${now}_${i}`,
      category,
      itemId,
      itemName,
      dataUrl: framesDataUrl[i],
      timestamp: now + i,
    };
    store.put(item);
  }

  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve(framesDataUrl.length);
    tx.onerror = () => reject(tx.error);
  });
}

/**
 * Obtiene el conteo total de frames para cada itemId
 */
export async function getAllFrameCounts(): Promise<Record<string, number>> {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readonly");
  const store = tx.objectStore(STORE_NAME);
  const request = store.getAll();

  return new Promise((resolve, reject) => {
    request.onsuccess = () => {
      const all = request.result as StoredFrame[];
      const counts: Record<string, number> = {};
      for (const item of all) {
        counts[item.itemId] = (counts[item.itemId] || 0) + 1;
      }
      resolve(counts);
    };
    request.onerror = () => reject(request.error);
  });
}

/**
 * Obtiene las imágenes (dataUrls) de un ítem para previsualización
 */
export async function getItemFrames(itemId: string, limit = 20): Promise<string[]> {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readonly");
  const store = tx.objectStore(STORE_NAME);
  const index = store.index("itemId");
  const request = index.getAll(itemId);

  return new Promise((resolve, reject) => {
    request.onsuccess = () => {
      const records = (request.result as StoredFrame[]).slice(-limit);
      resolve(records.map((r) => r.dataUrl));
    };
    request.onerror = () => reject(request.error);
  });
}

/**
 * Borra los frames de un ítem en específico
 */
export async function deleteItemFrames(itemId: string): Promise<void> {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);
  const index = store.index("itemId");
  const request = index.getAllKeys(itemId);

  return new Promise((resolve, reject) => {
    request.onsuccess = () => {
      const keys = request.result;
      for (const key of keys) {
        store.delete(key);
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/**
 * Exporta el dataset completo como archivo ZIP estructurado por carpetas:
 * dataset/
 *   letras_y_numeros/
 *     A/frame_001.jpg
 *   palabras/
 *     HOLA/frame_001.jpg
 */
export async function exportDatasetAsZip(onProgress?: (progress: number) => void): Promise<Blob> {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readonly");
  const store = tx.objectStore(STORE_NAME);
  const request = store.getAll();

  const allFrames: StoredFrame[] = await new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result as StoredFrame[]);
    request.onerror = () => reject(request.error);
  });

  if (allFrames.length === 0) {
    throw new Error("No hay fotos capturadas aún en el dataset");
  }

  const zip = new JSZip();
  const root = zip.folder("dataset");

  const itemIndexes: Record<string, number> = {};

  for (let i = 0; i < allFrames.length; i++) {
    const frame = allFrames[i];
    const categoryFolder = frame.category === "letras_y_numeros" ? "letras_y_numeros" : "palabras";
    const cleanItemName = frame.itemName.replace(/[^a-zA-Z0-9_ÑáéíóúÁÉÍÓÚ]/g, "_");

    itemIndexes[frame.itemId] = (itemIndexes[frame.itemId] || 0) + 1;
    const frameNum = String(itemIndexes[frame.itemId]).padStart(4, "0");
    const filename = `frame_${frameNum}.jpg`;

    // Convert dataUrl base64 to binary
    const base64Data = frame.dataUrl.split(",")[1];
    root?.folder(categoryFolder)?.folder(cleanItemName)?.file(filename, base64Data, { base64: true });

    if (onProgress && i % 10 === 0) {
      onProgress(Math.round((i / allFrames.length) * 50));
    }
  }

  // Generar zip binario
  return await zip.generateAsync({ type: "blob" }, (metadata) => {
    if (onProgress) {
      onProgress(50 + Math.round(metadata.percent * 0.5));
    }
  });
}
