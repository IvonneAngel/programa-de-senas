import React from "react";
import {
  Volume2,
  VolumeX,
  Video,
  VideoOff,
  Mic,
  MicOff,
  Phone,
  Settings as SettingsIcon,
  Smile,
  Hand,
} from "lucide-react";

interface DockProps {
  accentColor: string;
  menuExpanded: boolean;
  onToggleMenuExpansion: () => void;
  showStickerDrawer: boolean;
  onToggleStickerDrawer: () => void;
  panelActive: boolean;
  onToggleSettings: () => void;
  trainingActive: boolean;
  onToggleTraining: () => void;
  cameraActive: boolean;
  onToggleCamera: () => void;
  speakerActive: boolean;
  onToggleSpeaker: () => void;
  micActive: boolean;
  onToggleMic: () => void;
  callConnected: boolean;
  onToggleCall: () => void;
}

export default function Dock({
  accentColor,
  menuExpanded,
  onToggleMenuExpansion,
  showStickerDrawer,
  onToggleStickerDrawer,
  panelActive,
  onToggleSettings,
  trainingActive,
  onToggleTraining,
  cameraActive,
  onToggleCamera,
  speakerActive,
  onToggleSpeaker,
  micActive,
  onToggleMic,
  callConnected,
  onToggleCall,
}: DockProps) {
  return (
    <div className="flex items-end pointer-events-auto select-none">
      {/* Esquina curva izquierda */}
      <svg className="w-5 h-5 text-[#0a0a0a]" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M24 24V0C24 13.2548 13.2548 24 0 24H24Z" fill="currentColor" />
        <path
          d="M24 0C24 13.2548 13.2548 24 0 24"
          fill="none"
          stroke="rgba(255,255,255,0.04)"
          strokeWidth="1"
        />
      </svg>

      {/* Fondo del dock que aloja la barra de controles */}
      <div className="bg-[#0a0a0a] px-5 pt-1.5 pb-2.5 rounded-t-[28px] border-t border-x border-white/5 shadow-[0_-10px_25px_rgba(0,0,0,0.6)] flex items-center justify-center">
        {/* BARRA DE BOTONES (Se estira y contrae horizontalmente con CSS fluido) */}
        <div className="bg-[#181818]/90 backdrop-blur-md py-1.5 px-3.5 rounded-[100px] flex items-center gap-2 shadow-[0_15px_40px_rgba(0,0,0,0.8)] border border-white/5 transition-all duration-300">
          
          {/* Botón Ajustes (...) - Expansor principal */}
          <button
            id="btn-settings-toggle"
            type="button"
            onClick={onToggleMenuExpansion}
            className={`w-11 h-11 min-w-[44px] rounded-full border-0 outline-none flex justify-center items-center cursor-pointer transition-all active:scale-[0.9] shadow-[inset_0_2px_4px_rgba(0,0,0,0.8),0_1px_1px_rgba(255,255,255,0.05)] ${
              menuExpanded ? "bg-[#252525]" : "bg-[#101010] hover:bg-[#151515]"
            }`}
            title="Más controles"
          >
            <svg
              className="w-[18px] h-[18px] transition-colors duration-300"
              style={{ fill: accentColor }}
              viewBox="0 0 24 24"
            >
              <circle cx="5" cy="12" r="2" />
              <circle cx="12" cy="12" r="2" />
              <circle cx="19" cy="12" r="2" />
            </svg>
          </button>

          {/* Contenedor expandible fluido sin huecos fantasma en CSS */}
          <div
            className={`flex items-center overflow-hidden transition-all duration-300 ease-out ${
              menuExpanded
                ? "max-w-[110px] gap-2 opacity-100"
                : "max-w-0 gap-0 opacity-0 pointer-events-none"
            }`}
            style={{
              marginRight: menuExpanded ? 0 : "-8px",
            }}
          >
            {/* 1. Botón de Mano (Entrenamiento y Color) */}
            <button
              id="btn-training-toggle"
              type="button"
              onClick={onToggleTraining}
              className={`w-11 h-11 min-w-[44px] rounded-full border-0 outline-none flex justify-center items-center cursor-pointer transition-all active:scale-[0.9] shadow-[inset_0_2px_4px_rgba(0,0,0,0.8),0_1px_1px_rgba(255,255,255,0.05)] overflow-hidden ${
                trainingActive
                  ? "bg-emerald-600 text-black font-bold"
                  : "bg-[#101010] hover:bg-[#151515]"
              }`}
              title="Mano (Entrenamiento y Color)"
            >
              <Hand
                className={`w-[18px] h-[18px] transition-transform ${
                  trainingActive ? "text-white scale-110" : "text-neutral-400"
                }`}
              />
            </button>

            {/* 2. Botón de Stickers de Señales */}
            <button
              id="btn-stickers-toggle"
              type="button"
              onClick={onToggleStickerDrawer}
              className={`w-11 h-11 min-w-[44px] rounded-full border-0 outline-none flex justify-center items-center cursor-pointer transition-all active:scale-[0.9] shadow-[inset_0_2px_4px_rgba(0,0,0,0.8),0_1px_1px_rgba(255,255,255,0.05)] overflow-hidden ${
                showStickerDrawer
                  ? "bg-emerald-950 text-white"
                  : "bg-[#101010] hover:bg-[#151515]"
              }`}
              title="Mis Stickers"
            >
              <Smile
                className={`w-[18px] h-[18px] ${
                  showStickerDrawer ? "text-emerald-400" : "text-neutral-400"
                }`}
              />
            </button>
          </div>

          {/* Botón Cámara */}
          <button
            id="btn-camera-toggle"
            type="button"
            onClick={onToggleCamera}
            className="w-11 h-11 rounded-full border-0 outline-none flex justify-center items-center cursor-pointer bg-[#101010] hover:bg-[#151515] transition-all active:scale-[0.9] shadow-[inset_0_2px_4px_rgba(0,0,0,0.8),0_1px_1px_rgba(255,255,255,0.05)]"
            title={cameraActive ? "Desactivar cámara" : "Activar cámara"}
          >
            {cameraActive ? (
              <Video className="w-[18px] h-[18px] text-white" />
            ) : (
              <VideoOff className="w-[18px] h-[18px] text-neutral-500" />
            )}
          </button>

          {/* Botón Altavoz */}
          <button
            id="btn-speaker-toggle"
            type="button"
            onClick={onToggleSpeaker}
            className="w-11 h-11 rounded-full border-0 outline-none flex justify-center items-center cursor-pointer bg-[#101010] hover:bg-[#151515] transition-all active:scale-[0.9] shadow-[inset_0_2px_4px_rgba(0,0,0,0.8),0_1px_1px_rgba(255,255,255,0.05)]"
            title={speakerActive ? "Silenciar audio" : "Activar audio"}
          >
            {speakerActive ? (
              <Volume2 className="w-[18px] h-[18px] text-white" />
            ) : (
              <VolumeX className="w-[18px] h-[18px] text-neutral-500" />
            )}
          </button>

          {/* Botón Micrófono */}
          <button
            id="btn-mic-toggle"
            type="button"
            onClick={onToggleMic}
            className="w-11 h-11 rounded-full border-0 outline-none flex justify-center items-center cursor-pointer bg-[#101010] hover:bg-[#151515] transition-all active:scale-[0.9] shadow-[inset_0_2px_4px_rgba(0,0,0,0.8),0_1px_1px_rgba(255,255,255,0.05)]"
            title={micActive ? "Silenciar micrófono" : "Activar micrófono"}
          >
            {micActive ? (
              <Mic className="w-[18px] h-[18px] text-white" />
            ) : (
              <MicOff className="w-[18px] h-[18px] text-neutral-500" />
            )}
          </button>

          {/* Botón Colgar / Conectar */}
          <button
            id="btn-call-action"
            type="button"
            onClick={onToggleCall}
            className={`w-[50px] h-[50px] rounded-full border-0 outline-none flex justify-center items-center cursor-pointer transition-all active:scale-[0.9] ${
              callConnected
                ? "bg-[#ff124e] shadow-[inset_0_2px_6px_rgba(255,255,255,0.3),0_6px_15px_rgba(255,18,78,0.35)] hover:brightness-110"
                : "bg-emerald-600 shadow-[inset_0_2px_6px_rgba(255,255,255,0.3),0_6px_15px_rgba(16,185,129,0.35)] hover:brightness-110"
            }`}
            title={callConnected ? "Desconectar llamada" : "Conectar llamada"}
          >
            <Phone
              className={`w-6 h-6 text-white transition-transform duration-300 ${
                callConnected ? "rotate-[135deg]" : "rotate-0"
              }`}
            />
          </button>
        </div>
      </div>

      {/* Esquina curva derecha */}
      <svg
        className="w-5 h-5 text-[#0a0a0a]"
        style={{ transform: "scaleX(-1)" }}
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path d="M24 24V0C24 13.2548 13.2548 24 0 24H24Z" fill="currentColor" />
        <path
          d="M24 0C24 13.2548 13.2548 24 0 24"
          fill="none"
          stroke="rgba(255,255,255,0.04)"
          strokeWidth="1"
        />
      </svg>
    </div>
  );
}
