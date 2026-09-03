import React from "react";
import { FloatingEmoji } from "../types";

interface FloatingReactionsProps {
  reactions: FloatingEmoji[];
  onRemoveReaction: (id: number) => void;
}

export default function FloatingReactions({ reactions, onRemoveReaction }: FloatingReactionsProps) {
  if (reactions.length === 0) return null;

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden z-50">
      {reactions.map((r) => {
        const shiftX = (Math.random() * 80 - 40).toFixed(0);
        const rotStart = `${r.rotation}deg`;
        const rotMid = `${r.rotation + (Math.random() * 30 - 15)}deg`;
        const rotEnd = `${r.rotation + (Math.random() * 50 - 25)}deg`;

        return (
          <div
            key={r.id}
            onAnimationEnd={() => onRemoveReaction(r.id)}
            className="reaction-floating-item filter drop-shadow-[0_20px_40px_rgba(0,0,0,0.8)]"
            style={
              {
                left: `${r.x}%`,
                "--shift-x": `${shiftX}px`,
                "--rot-start": rotStart,
                "--rot-mid": rotMid,
                "--rot-end": rotEnd,
                "--target-scale": r.scale,
              } as React.CSSProperties
            }
          >
            {r.imageUrl ? (
              <img
                src={r.imageUrl}
                alt={r.text || "sticker"}
                className="w-80 h-80 object-contain"
                style={{ filter: "url(#remove-white)" }}
                referrerPolicy="no-referrer"
              />
            ) : (
              <span className="text-8xl select-none">{r.emoji}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
