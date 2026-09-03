import React from "react";
import { STICKER_TEMPLATES } from "../data/stickers";
import { StickerTemplate } from "../types";

interface StickerDrawerProps {
  isOpen: boolean;
  onSelectSticker: (template: StickerTemplate) => void;
}

export default function StickerDrawer({ isOpen, onSelectSticker }: StickerDrawerProps) {
  return (
    <div
      className={`floating-panel-spring w-[min(440px,calc(100vw-36px))] bg-[#121212]/95 backdrop-blur-md p-3 rounded-[32px] shadow-[0_25px_50px_rgba(0,0,0,0.85),inset_0_1px_1px_rgba(255,255,255,0.05)] border border-white/5 sticker-picker-panel overflow-hidden ${
        isOpen ? "floating-panel-visible" : "floating-panel-hidden"
      }`}
    >
      <div className="grid grid-cols-2 gap-4 max-h-[320px] overflow-y-auto pr-1 pb-1 scrollbar-slim">
        {STICKER_TEMPLATES.map((temp) => (
          <button
            key={temp.id}
            type="button"
            onClick={() => onSelectSticker(temp)}
            className="group relative h-[150px] bg-neutral-900/40 hover:bg-neutral-800/60 border border-white/5 hover:border-white/10 rounded-2xl flex items-center justify-center p-2 cursor-pointer transition-all active:scale-95 overflow-hidden shadow-inner outline-none"
          >
            {temp.imageUrl && (
              <img
                src={temp.imageUrl}
                alt={temp.text || "sticker"}
                className="w-36 h-36 object-contain z-10 filter drop-shadow-[0_6px_12px_rgba(0,0,0,0.5)] transform group-hover:scale-115 transition-transform duration-200"
                style={{ filter: "url(#remove-white)" }}
                referrerPolicy="no-referrer"
              />
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
