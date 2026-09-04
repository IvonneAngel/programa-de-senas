import React, { useEffect, useRef, useState } from "react";
import GhostWord from "./GhostWord";

type Props = { letra: string | null };

const STORAGE_KEY = "auroraBallPos_v2";
const DEFAULT_POS = { x: 16, y: 16 }; // top-4 left-4

// Esfera Aurora en iframe aislado (React 18 + WebGL) + arrastre con persistencia.
export default function AuroraBallWithLetter({ letra }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(DEFAULT_POS);
  const dragRef = useRef<{ offsetX: number; offsetY: number; dragging: boolean } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // cargar posición guardada
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const p = JSON.parse(raw);
        if (typeof p.x === "number" && typeof p.y === "number") setPos(p);
      }
    } catch {}
  }, []);

  const savePos = (p: { x: number; y: number }) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
    } catch {}
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const el = containerRef.current;
    const parent = el?.parentElement;
    if (!el || !parent) return;
    // no arrastrar si es doble click para reset
    e.preventDefault();
    e.stopPropagation();
    const parentRect = parent.getBoundingClientRect();
    const elRect = el.getBoundingClientRect();
    dragRef.current = {
      offsetX: e.clientX - elRect.left,
      offsetY: e.clientY - elRect.top,
      dragging: true,
    };
    setIsDragging(true);
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!dragRef.current?.dragging) return;
      const el = containerRef.current;
      const parent = el?.parentElement;
      if (!el || !parent) return;
      const parentRect = parent.getBoundingClientRect();
      const elRect = el.getBoundingClientRect();
      let x = e.clientX - parentRect.left - dragRef.current.offsetX;
      let y = e.clientY - parentRect.top - dragRef.current.offsetY;
      x = Math.max(0, Math.min(x, parentRect.width - elRect.width));
      y = Math.max(0, Math.min(y, parentRect.height - elRect.height));
      setPos({ x, y });
    };
    const onUp = () => {
      if (dragRef.current?.dragging) {
        dragRef.current.dragging = false;
        setIsDragging(false);
      }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, []);

  // guardar persistencia cuando termina drag o cambia pos en reposo
  useEffect(() => {
    if (!isDragging) savePos(pos);
  }, [pos, isDragging]);

  const onDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPos(DEFAULT_POS);
    savePos(DEFAULT_POS);
  };

  // si letra es palabra completa (ej "hola"), la muestra tal cual; si es 1 letra, GhostWord añade fantasma
  const letraNorm = letra?.trim() || null;
  const mostrarGhost = letraNorm && letraNorm.length === 1;

  return (
    <div
      ref={containerRef}
      className={`absolute z-20 flex items-center gap-2 select-none ${isDragging ? "cursor-grabbing" : "cursor-grab"}`}
      style={{ left: pos.x, top: pos.y, touchAction: "none" }}
      onPointerDown={onPointerDown}
      onDoubleClick={onDoubleClick}
      title="Arrastra para mover — doble click para reset"
      aria-label="Aurora ball arrastrable"
    >
      <div className="w-7 h-7 rounded-full shrink-0 overflow-hidden block relative">
        <iframe
          src="/aurora-ball.html?v=color-5min-1"
          title="Aurora"
          aria-label="Aurora ball"
          className="w-7 h-7 rounded-full block overflow-hidden"
          style={{ border: "none", display: "block", background: "transparent", pointerEvents: "none" }}
          scrolling="no"
          loading="eager"
          tabIndex={-1}
        />
        {/* overlay transparente para capturar drag sobre iframe */}
        <div className="absolute inset-0" aria-hidden="true" />
      </div>
      {letraNorm ? (
        <span
          className="text-white/95 text-[13px] font-medium tracking-wide px-2 py-1 rounded-full bg-black/60 backdrop-blur-md border border-white/10 max-w-[180px] truncate"
          style={{ fontFamily: '"Sheriff Sans", sans-serif' }}
        >
          {letraNorm}
          {mostrarGhost ? <GhostWord letra={letraNorm} /> : null}
        </span>
      ) : null}
    </div>
  );
}
