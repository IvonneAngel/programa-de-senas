import React, { useState, useRef, useEffect, useCallback } from "react";
import { FloatingEmoji, StickerTemplate } from "./types";
import { playAudioFeedback } from "./utils/audio";
import CameraFeed from "./components/CameraFeed";
import FloatingReactions from "./components/FloatingReactions";
import StickerDrawer from "./components/StickerDrawer";
import SettingsPanel from "./components/SettingsPanel";
import TrainingPanel from "./components/TrainingPanel";
import Dock from "./components/Dock";

export default function App() {
  const [panelActive, setPanelActive] = useState(false);
  const [accentColor, setAccentColor] = useState("hsl(180, 100%, 50%)");

  // Call & device states
  const [cameraActive, setCameraActive] = useState(false);
  const [speakerActive, setSpeakerActive] = useState(true);
  const [micActive, setMicActive] = useState(false);
  const [callConnected, setCallConnected] = useState(true);

  // Menu expansion state
  const [menuExpanded, setMenuExpanded] = useState(false);
  const [showStickerDrawer, setShowStickerDrawer] = useState(false);
  const [showTrainingPanel, setShowTrainingPanel] = useState(false);

  // Floating reactions list
  const [reactions, setReactions] = useState<FloatingEmoji[]>([]);

  // Controls visibility toggle (via double click/tap)
  const [controlsVisible, setControlsVisible] = useState(true);

  // Refs for media stream and touch handling
  const mainDisplayRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastTapRef = useRef<number>(0);

  // Menu toggles with instant audio feedback
  const toggleMenuExpansion = useCallback(() => {
    setMenuExpanded((prev) => {
      const next = !prev;
      if (!next) {
        setPanelActive(false);
        setShowStickerDrawer(false);
        setShowTrainingPanel(false);
      }
      return next;
    });
    playAudioFeedback("pop", speakerActive);
  }, [speakerActive]);

  const toggleSettings = useCallback(() => {
    setPanelActive((prev) => !prev);
    setShowStickerDrawer(false);
    setShowTrainingPanel(false);
    playAudioFeedback("pop", speakerActive);
  }, [speakerActive]);

  const toggleStickerDrawer = useCallback(() => {
    setShowStickerDrawer((prev) => !prev);
    setPanelActive(false);
    setShowTrainingPanel(false);
    playAudioFeedback("pop", speakerActive);
  }, [speakerActive]);

  const toggleTraining = useCallback(() => {
    setShowTrainingPanel((prev) => !prev);
    setPanelActive(false);
    setShowStickerDrawer(false);
    playAudioFeedback("pop", speakerActive);
  }, [speakerActive]);

  const triggerReaction = useCallback((template: StickerTemplate) => {
    const id = Date.now() + Math.random();
    const newEmoji: FloatingEmoji = {
      id,
      emoji: template.emoji,
      x: 15 + Math.random() * 70, // Start position percentage (15% to 85%)
      scale: 0.9 + Math.random() * 0.4,
      rotation: (Math.random() - 0.5) * 40,
      text: template.text,
      splashPath: template.splashPath,
      splashGradient: template.splashGradient,
      textStyle: template.textStyle,
      imageUrl: template.imageUrl,
    };
    setReactions((prev) => [...prev, newEmoji]);
    playAudioFeedback("bubble", speakerActive);
  }, [speakerActive]);

  const handleRemoveReaction = useCallback((id: number) => {
    setReactions((prev) => prev.filter((r) => r.id !== id));
  }, []);

  const handleToggleCamera = useCallback(() => {
    setCameraActive((prev) => !prev);
    playAudioFeedback("pop", speakerActive);
  }, [speakerActive]);

  const handleToggleSpeaker = useCallback(() => {
    setSpeakerActive((prev) => {
      const next = !prev;
      if (next) {
        playAudioFeedback("chime", true);
      }
      return next;
    });
  }, []);

  const handleToggleMic = useCallback(() => {
    setMicActive((prev) => !prev);
    playAudioFeedback("pop", speakerActive);
  }, [speakerActive]);

  const handleToggleCall = useCallback(() => {
    if (callConnected) {
      playAudioFeedback("end", speakerActive);
      setCallConnected(false);
      setCameraActive(false);
    } else {
      setCallConnected(true);
      playAudioFeedback("pop", speakerActive);
    }
  }, [callConnected, speakerActive]);

  // Global double-click & double-tap to toggle controls visibility
  const handleDoubleClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (
      target.closest("button") ||
      target.closest(".settings-panel") ||
      target.closest(".sticker-picker-panel") ||
      target.closest(".no-double-click")
    ) {
      return;
    }
    setControlsVisible((prev) => !prev);
    playAudioFeedback("pop", speakerActive);
  };

  const handleTouchStartGlobal = (e: React.TouchEvent) => {
    const target = e.target as HTMLElement;
    if (
      target.closest("button") ||
      target.closest(".settings-panel") ||
      target.closest(".sticker-picker-panel") ||
      target.closest(".no-double-click")
    ) {
      return;
    }
    const now = Date.now();
    const DOUBLE_TAP_DELAY = 300;
    if (now - lastTapRef.current < DOUBLE_TAP_DELAY) {
      setControlsVisible((prev) => !prev);
      playAudioFeedback("pop", speakerActive);
      lastTapRef.current = 0;
    } else {
      lastTapRef.current = now;
    }
  };

  // Optimized camera stream lifecycle
  useEffect(() => {
    let active = true;
    if (cameraActive && callConnected) {
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
        },
      };

      navigator.mediaDevices
        ?.getUserMedia(constraints)
        .then((stream) => {
          if (active) {
            streamRef.current = stream;
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
            }
          } else {
            stream.getTracks().forEach((track) => track.stop());
          }
        })
        .catch((err) => {
          console.warn("Could not access camera (preview fallback active):", err);
        });
    } else {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }

    return () => {
      active = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, [cameraActive, callConnected]);

  return (
    <div
      onDoubleClick={handleDoubleClick}
      onTouchStart={handleTouchStartGlobal}
      id="app-root-container"
      className="h-screen w-screen bg-[#0a0a0a] bg-[radial-gradient(#1c1c1c_1px,transparent_1px)] bg-[size:25px_25px] font-sans overflow-hidden flex flex-col justify-between items-center p-2 md:p-3 pb-3 select-none"
    >
      {/* SVG filter to remove white background from JPEG stickers */}
      <svg
        style={{ position: "absolute", width: 0, height: 0, overflow: "hidden" }}
        aria-hidden="true"
      >
        <defs>
          <filter id="remove-white" colorInterpolationFilters="sRGB">
            <feColorMatrix
              type="matrix"
              values="
              1 0 0 0 0
              0 1 0 0 0
              0 0 1 0 0
              -1.8 -1.8 -1.8 5.4 -0.05
            "
            />
          </filter>
        </defs>
      </svg>

      {/* PANTALLA PRINCIPAL (Ocupa todo el ancho, curvada como el iPhone) */}
      <div
        ref={mainDisplayRef}
        id="main-display-screen"
        className="w-full flex-1 bg-[#0b0b0b] rounded-[36px] md:rounded-[44px] border border-white/5 overflow-hidden relative shadow-[0_25px_60px_-15px_rgba(0,0,0,0.9)] flex items-center justify-center transition-all duration-300"
      >
        {/* REACCIONES FLOTANTES (Animadas con GPU en CSS puro) */}
        <FloatingReactions
          reactions={reactions}
          onRemoveReaction={handleRemoveReaction}
        />

        {/* FEED DE CÁMARA */}
        <CameraFeed
          videoRef={videoRef}
          cameraActive={cameraActive}
          callConnected={callConnected}
        />

        {/* INDICADOR SUTIL DE DOBLE TOQUE */}
        <div
          className={`indicator-fade absolute top-4 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/5 text-[10px] font-medium tracking-widest text-neutral-400 select-none pointer-events-none uppercase ${
            controlsVisible
              ? "opacity-0 -translate-y-2 pointer-events-none"
              : "opacity-60 translate-y-0"
          }`}
        >
          Doble toque para mostrar controles
        </div>

        {/* CONTENEDOR DE LOS CONTROLES Y MENÚS FLOTANTES (Perfectamente centrado en X con deslizamiento fluido en Y) */}
        <div
          className={`absolute bottom-0 left-1/2 -translate-x-1/2 z-30 flex flex-col items-center pointer-events-none transition-transform duration-300 ease-out ${
            controlsVisible ? "translate-y-0" : "translate-y-[140%]"
          }`}
        >
          {/* SECCIÓN FLOTANTE DE AJUSTES, STICKERS Y ENTRENAMIENTO */}
          {(showStickerDrawer || panelActive || showTrainingPanel) && (
            <div className="flex flex-col items-center mb-2 pointer-events-none">
              {showStickerDrawer && (
                <StickerDrawer
                  isOpen={showStickerDrawer}
                  onSelectSticker={triggerReaction}
                />
              )}

              {panelActive && (
                <SettingsPanel
                  isOpen={panelActive}
                  accentColor={accentColor}
                  onColorChange={setAccentColor}
                />
              )}

              {showTrainingPanel && (
                <TrainingPanel
                  isOpen={showTrainingPanel}
                  onClose={() => setShowTrainingPanel(false)}
                  videoRef={videoRef}
                  cameraActive={cameraActive}
                  onActivateCamera={() => setCameraActive(true)}
                  speakerActive={speakerActive}
                  accentColor={accentColor}
                  onColorChange={setAccentColor}
                />
              )}
            </div>
          )}

          {/* DOCK CURVADO */}
          <Dock
            accentColor={accentColor}
            menuExpanded={menuExpanded}
            onToggleMenuExpansion={toggleMenuExpansion}
            showStickerDrawer={showStickerDrawer}
            onToggleStickerDrawer={toggleStickerDrawer}
            panelActive={panelActive}
            onToggleSettings={toggleSettings}
            trainingActive={showTrainingPanel}
            onToggleTraining={toggleTraining}
            cameraActive={cameraActive}
            onToggleCamera={handleToggleCamera}
            speakerActive={speakerActive}
            onToggleSpeaker={handleToggleSpeaker}
            micActive={micActive}
            onToggleMic={handleToggleMic}
            callConnected={callConnected}
            onToggleCall={handleToggleCall}
          />
        </div>
      </div>
    </div>
  );
}
