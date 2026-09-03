import React from "react";
import { VideoOff } from "lucide-react";

interface CameraFeedProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraActive: boolean;
  callConnected: boolean;
}

export default function CameraFeed({
  videoRef,
  cameraActive,
  callConnected,
}: CameraFeedProps) {
  if (cameraActive && callConnected) {
    return (
      <div className="absolute inset-0 w-full h-full bg-black overflow-hidden flex items-center justify-center">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover scale-x-[-1]"
        />
        {/* Subtle ambient gradient vignette */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/20 pointer-events-none" />
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center select-none p-8 z-10 text-center max-w-sm">
      <div className="w-20 h-20 rounded-full bg-neutral-900/60 flex items-center justify-center border border-white/5 shadow-[0_8px_30px_rgb(0,0,0,0.6)] mb-4 animate-pulse">
        <VideoOff className="w-9 h-9 text-neutral-400" />
      </div>
      <h3 className="text-white/95 font-bold tracking-tight text-base mb-1 font-sans">
        Cámara Desactivada
      </h3>
      <p className="text-xs text-neutral-500 max-w-[280px]">
        {callConnected
          ? "Activa la cámara con el botón del dock para empezar a colocar stickers interactivos en tu feed"
          : "Conecta la llamada para iniciar la cámara."}
      </p>
    </div>
  );
}
