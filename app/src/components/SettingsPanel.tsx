import React from "react";
import ColorWheel from "./ColorWheel";

interface SettingsPanelProps {
  isOpen: boolean;
  accentColor: string;
  onColorChange: (color: string) => void;
}

export default function SettingsPanel({
  isOpen,
  accentColor,
  onColorChange,
}: SettingsPanelProps) {
  return (
    <div
      className={`floating-panel-spring w-[260px] bg-[#141414]/95 backdrop-blur-md p-4 rounded-[32px] shadow-[0_25px_50px_rgba(0,0,0,0.85),inset_0_1px_1px_rgba(255,255,255,0.05)] border border-white/5 flex flex-col items-center justify-center settings-panel ${
        isOpen ? "floating-panel-visible" : "floating-panel-hidden"
      }`}
    >
      {/* Rueda de color interactiva sin textos */}
      <ColorWheel initialColor={accentColor} onColorChange={onColorChange} />
    </div>
  );
}

