import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Download,
  Plus,
  Trash2,
  Square,
  Camera,
  Check,
  X,
} from "lucide-react";
import { DatasetCategory, DatasetItem } from "../types";
import {
  DEFAULT_LETRAS,
  DEFAULT_NUMEROS,
  DEFAULT_PALABRAS,
  getCustomWords,
  saveCustomWord,
  deleteCustomWord,
} from "../training/categories";
import {
  saveItemFrames,
  getAllFrameCounts,
  deleteItemFrames,
  exportDatasetAsZip,
} from "../training/datasetStorage";
import { playAudioFeedback } from "../utils/audio";
import ColorWheel from "./ColorWheel";

type PanelTab = DatasetCategory | "color";

interface TrainingPanelProps {
  isOpen: boolean;
  onClose: () => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraActive: boolean;
  onActivateCamera: () => void;
  speakerActive: boolean;
  accentColor?: string;
  onColorChange?: (color: string) => void;
}

export default function TrainingPanel({
  isOpen,
  videoRef,
  cameraActive,
  onActivateCamera,
  speakerActive,
  accentColor = "#00ffff",
  onColorChange = () => {},
}: TrainingPanelProps) {
  // Tabs: letras_y_numeros | palabras | color
  const [activeTab, setActiveTab] = useState<PanelTab>("letras_y_numeros");

  // Lista de items
  const [customWords, setCustomWords] = useState<DatasetItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<DatasetItem>(DEFAULT_LETRAS[0]);

  // Conteos
  const [frameCounts, setFrameCounts] = useState<Record<string, number>>({});

  // Cantidad de frames: se ingresa como número directo
  const [targetFrames, setTargetFrames] = useState<number>(30);

  // Estados de captura
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [capturedCurrent, setCapturedCurrent] = useState<number>(0);

  // Modo creación de palabra rápida (inline, sin modal)
  const [isAddingWord, setIsAddingWord] = useState<boolean>(false);
  const [newWordText, setNewWordText] = useState<string>("");

  // Exportar ZIP
  const [isExporting, setIsExporting] = useState<boolean>(false);

  const captureAbortRef = useRef<boolean>(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const refreshDataset = useCallback(async () => {
    try {
      const counts = await getAllFrameCounts();
      setFrameCounts(counts);
      setCustomWords(getCustomWords());
    } catch (err) {
      console.error("Error al refrescar dataset:", err);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      refreshDataset();
    }
  }, [isOpen, refreshDataset]);

  // Añadir palabra nueva inline
  const handleSaveWord = () => {
    if (!newWordText.trim()) {
      setIsAddingWord(false);
      return;
    }
    const created = saveCustomWord(newWordText);
    if (created) {
      playAudioFeedback("pop", speakerActive);
      setNewWordText("");
      setIsAddingWord(false);
      refreshDataset();
      setSelectedItem(created);
    } else {
      setIsAddingWord(false);
    }
  };

  // Borrar palabra personalizada
  const handleDeleteCustomWord = async (e: React.MouseEvent, item: DatasetItem) => {
    e.stopPropagation();
    deleteCustomWord(item.id);
    await deleteItemFrames(item.id);
    playAudioFeedback("end", speakerActive);
    refreshDataset();
    if (selectedItem?.id === item.id) {
      setSelectedItem(DEFAULT_PALABRAS[0]);
    }
  };

  // Capturar frame desde el feed de video
  const captureFrame = (): string | null => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return null;

    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }
    const canvas = canvasRef.current;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.save();
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, width, height);
    ctx.restore();

    return canvas.toDataURL("image/jpeg", 0.82);
  };

  // Tomar ráfaga
  const handleCaptureBurst = async () => {
    if (!cameraActive) {
      onActivateCamera();
      return;
    }
    if (!selectedItem) return;

    const framesToTake = Math.max(1, Math.min(500, targetFrames || 30));
    captureAbortRef.current = false;
    setIsCapturing(true);
    setCapturedCurrent(0);

    // Cuenta regresiva 3, 2, 1
    for (let c = 3; c > 0; c--) {
      if (captureAbortRef.current) {
        setIsCapturing(false);
        setCountdown(null);
        return;
      }
      setCountdown(c);
      playAudioFeedback("chime", speakerActive);
      await new Promise((r) => setTimeout(r, 800));
    }

    setCountdown(null);

    const captured: string[] = [];
    const intervalMs = 110;

    for (let i = 1; i <= framesToTake; i++) {
      if (captureAbortRef.current) break;

      const frameData = captureFrame();
      if (frameData) {
        captured.push(frameData);
        setCapturedCurrent(i);
        if (i % 4 === 0) {
          playAudioFeedback("bubble", speakerActive);
        }
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }

    if (captured.length > 0) {
      await saveItemFrames(
        selectedItem.category,
        selectedItem.id,
        selectedItem.name,
        captured
      );
      playAudioFeedback("pop", speakerActive);
      await refreshDataset();
    }

    setIsCapturing(false);
    setCountdown(null);
    setCapturedCurrent(0);
  };

  const handleCancelCapture = () => {
    captureAbortRef.current = true;
    setIsCapturing(false);
    setCountdown(null);
    setCapturedCurrent(0);
  };

  const handleExportZip = async () => {
    try {
      setIsExporting(true);
      const zipBlob = await exportDatasetAsZip();
      const url = URL.createObjectURL(zipBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `dataset_${Date.now()}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      playAudioFeedback("pop", speakerActive);
    } catch {
      // Ignorar si está vacío
    } finally {
      setIsExporting(false);
    }
  };

  const totalFrames: number = Object.values(frameCounts).reduce<number>(
    (acc, val) => acc + (typeof val === "number" ? val : 0),
    0
  );

  const currentItemCount = selectedItem ? frameCounts[selectedItem.id] || 0 : 0;

  return (
    <div
      className={`floating-panel-spring w-[min(340px,calc(100vw-32px))] bg-[#141414]/95 backdrop-blur-md p-3.5 rounded-[32px] shadow-[0_25px_50px_rgba(0,0,0,0.85),inset_0_1px_1px_rgba(255,255,255,0.05)] border border-white/5 flex flex-col gap-2.5 pointer-events-auto ${
        isOpen ? "floating-panel-visible" : "floating-panel-hidden"
      }`}
    >
      {/* CABECERA MINIMALISTA */}
      <div className="flex justify-between items-center text-[10px] tracking-widest text-neutral-500 font-semibold select-none px-1">
        <span>ENTRENAMIENTO</span>
        {totalFrames > 0 && (
          <button
            type="button"
            onClick={handleExportZip}
            disabled={isExporting}
            className="flex items-center gap-1 text-[9px] text-neutral-400 hover:text-white transition-colors cursor-pointer"
            title="Descargar ZIP"
          >
            <Download className="w-3 h-3 text-emerald-400" />
            <span>{totalFrames}</span>
          </button>
        )}
      </div>

      {/* TABS DE LAS CATEGORÍAS Y COLOR */}
      <div className="grid grid-cols-3 gap-1 p-1 bg-[#0d0d0d] rounded-2xl border border-white/5">
        <button
          type="button"
          onClick={() => {
            setActiveTab("letras_y_numeros");
            setSelectedItem(DEFAULT_LETRAS[0]);
            playAudioFeedback("pop", speakerActive);
          }}
          className={`py-1.5 rounded-xl text-[10px] font-bold tracking-wider uppercase transition-all cursor-pointer ${
            activeTab === "letras_y_numeros"
              ? "bg-[#242424] text-white shadow-sm"
              : "text-neutral-500 hover:text-neutral-300"
          }`}
        >
          Letras
        </button>
        <button
          type="button"
          onClick={() => {
            setActiveTab("palabras");
            const words = [...DEFAULT_PALABRAS, ...customWords];
            setSelectedItem(words[0]);
            playAudioFeedback("pop", speakerActive);
          }}
          className={`py-1.5 rounded-xl text-[10px] font-bold tracking-wider uppercase transition-all cursor-pointer ${
            activeTab === "palabras"
              ? "bg-[#242424] text-white shadow-sm"
              : "text-neutral-500 hover:text-neutral-300"
          }`}
        >
          Palabras
        </button>
        <button
          type="button"
          onClick={() => {
            setActiveTab("color");
            playAudioFeedback("pop", speakerActive);
          }}
          className={`py-1.5 rounded-xl text-[10px] font-bold tracking-wider uppercase transition-all cursor-pointer flex items-center justify-center ${
            activeTab === "color"
              ? "bg-[#242424] text-white shadow-sm"
              : "text-neutral-500 hover:text-neutral-300"
          }`}
        >
          Color
        </button>
      </div>

      {activeTab === "color" ? (
        /* VISTA COLOR: RUEDA PURA SIN TEXTOS */
        <div className="flex flex-col items-center justify-center py-2">
          <ColorWheel initialColor={accentColor} onColorChange={onColorChange} />
        </div>
      ) : (
        <>
          {/* CONTENIDO CATEGORÍA 1: LETRAS Y NÚMEROS */}
          {activeTab === "letras_y_numeros" ? (
            <div className="flex flex-col gap-2 max-h-[165px] overflow-y-auto scrollbar-none">
              {/* 27 Letras */}
              <div className="grid grid-cols-7 gap-1">
                {DEFAULT_LETRAS.map((item) => {
                  const isSel = selectedItem?.id === item.id;
                  const count = frameCounts[item.id] || 0;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => {
                        setSelectedItem(item);
                        playAudioFeedback("pop", speakerActive);
                      }}
                      className={`h-7 rounded-lg text-[11px] font-bold transition-all active:scale-95 cursor-pointer relative flex items-center justify-center border ${
                        isSel
                          ? "bg-white text-black border-white shadow-sm"
                          : count > 0
                          ? "bg-[#1c1c1c] text-emerald-400 border-emerald-500/30"
                          : "bg-[#101010] text-neutral-400 border-white/5 hover:bg-[#181818]"
                      }`}
                    >
                      {item.name}
                      {count > 0 && !isSel && (
                        <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Números 0-9 */}
              <div className="grid grid-cols-5 gap-1 pt-1 border-t border-white/5">
                {DEFAULT_NUMEROS.map((item) => {
                  const isSel = selectedItem?.id === item.id;
                  const count = frameCounts[item.id] || 0;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => {
                        setSelectedItem(item);
                        playAudioFeedback("pop", speakerActive);
                      }}
                      className={`h-7 rounded-lg text-[11px] font-mono font-bold transition-all active:scale-95 cursor-pointer relative flex items-center justify-center border ${
                        isSel
                          ? "bg-white text-black border-white shadow-sm"
                          : count > 0
                          ? "bg-[#1c1c1c] text-emerald-400 border-emerald-500/30"
                          : "bg-[#101010] text-neutral-400 border-white/5 hover:bg-[#181818]"
                      }`}
                    >
                      {item.name}
                      {count > 0 && !isSel && (
                        <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            /* CONTENIDO CATEGORÍA 2: PALABRAS */
            <div className="flex flex-col gap-2 max-h-[165px] overflow-y-auto scrollbar-none">
              {/* Barra para añadir palabra rápida */}
              {isAddingWord ? (
                <div className="flex items-center gap-1 bg-[#101010] p-1 rounded-xl border border-white/10">
                  <input
                    type="text"
                    autoFocus
                    placeholder="NUEVA PALABRA..."
                    value={newWordText}
                    onChange={(e) => setNewWordText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleSaveWord();
                      if (e.key === "Escape") setIsAddingWord(false);
                    }}
                    className="w-full bg-transparent px-2 text-[10px] text-white uppercase font-bold focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={handleSaveWord}
                    className="w-6 h-6 rounded-lg bg-emerald-500 text-black flex items-center justify-center cursor-pointer"
                  >
                    <Check className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsAddingWord(false)}
                    className="w-6 h-6 rounded-lg bg-[#222] text-neutral-400 flex items-center justify-center cursor-pointer"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => {
                    setIsAddingWord(true);
                    playAudioFeedback("pop", speakerActive);
                  }}
                  className="py-1.5 px-2 rounded-xl bg-[#101010] hover:bg-[#1a1a1a] border border-white/5 text-[10px] font-bold text-neutral-400 hover:text-white flex items-center justify-center gap-1 transition-all cursor-pointer"
                >
                  <Plus className="w-3 h-3" />
                  AÑADIR PALABRA
                </button>
              )}

              {/* Lista de palabras en chips */}
              <div className="grid grid-cols-2 gap-1.5">
                {[...DEFAULT_PALABRAS, ...customWords].map((item) => {
                  const isSel = selectedItem?.id === item.id;
                  const count = frameCounts[item.id] || 0;
                  return (
                    <div
                      key={item.id}
                      onClick={() => {
                        setSelectedItem(item);
                        playAudioFeedback("pop", speakerActive);
                      }}
                      className={`h-8 px-2 rounded-xl flex items-center justify-between text-[10px] font-bold transition-all active:scale-98 cursor-pointer border select-none ${
                        isSel
                          ? "bg-white text-black border-white shadow-sm"
                          : count > 0
                          ? "bg-[#1c1c1c] text-emerald-400 border-emerald-500/30"
                          : "bg-[#101010] text-neutral-300 border-white/5 hover:bg-[#181818]"
                      }`}
                    >
                      <span className="truncate">{item.name}</span>
                      <div className="flex items-center gap-1">
                        {count > 0 && (
                          <span
                            className={`text-[8px] font-mono ${
                              isSel ? "text-neutral-700" : "text-emerald-400"
                            }`}
                          >
                            {count}
                          </span>
                        )}
                        {item.custom && (
                          <button
                            type="button"
                            onClick={(e) => handleDeleteCustomWord(e, item)}
                            className={`p-0.5 rounded hover:text-red-400 ${
                              isSel ? "text-neutral-500" : "text-neutral-600"
                            }`}
                          >
                            <Trash2 className="w-2.5 h-2.5" />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* CONFIGURACIÓN DE NÚMERO DE FRAMES (SIN BARRA DESLIZANTE, INGRESO DIRECTO) */}
          <div className="flex items-center justify-between bg-[#101010] px-3 py-1.5 rounded-2xl border border-white/5">
            <span className="text-[10px] text-neutral-400 font-bold uppercase tracking-wider select-none">
              Frames a capturar:
            </span>
            <div className="flex items-center gap-1.5">
              <input
                type="number"
                min={1}
                max={500}
                value={targetFrames === 0 ? "" : targetFrames}
                onChange={(e) => {
                  const val = parseInt(e.target.value, 10);
                  setTargetFrames(isNaN(val) ? 0 : val);
                }}
                className="no-spinners w-12 bg-[#1b1b1b] border border-white/10 rounded-lg py-1 px-1 text-center text-xs font-mono font-bold text-white focus:outline-none focus:border-white/30"
              />
            </div>
          </div>

          {/* BOTÓN DE DISPARO COMPACTO */}
          <div className="pt-0.5">
            {!isCapturing ? (
              <button
                type="button"
                onClick={handleCaptureBurst}
                className="w-full h-10 rounded-2xl bg-white text-black hover:bg-neutral-200 active:scale-95 font-bold text-xs tracking-wider uppercase flex items-center justify-center gap-1.5 transition-all shadow-md cursor-pointer select-none"
              >
                <Camera className="w-3.5 h-3.5" />
                <span>
                  {selectedItem
                    ? `Capturar ${selectedItem.name} (${currentItemCount})`
                    : "Capturar"}
                </span>
              </button>
            ) : (
              <button
                type="button"
                onClick={handleCancelCapture}
                className="w-full h-10 rounded-2xl bg-red-600 hover:bg-red-500 active:scale-95 text-white font-bold text-xs tracking-wider uppercase flex items-center justify-center gap-1.5 transition-all shadow-md cursor-pointer select-none"
              >
                <Square className="w-3.5 h-3.5 fill-white" />
                <span>
                  {countdown !== null
                    ? `Iniciando en ${countdown}...`
                    : `${capturedCurrent} / ${targetFrames} frames`}
                </span>
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
