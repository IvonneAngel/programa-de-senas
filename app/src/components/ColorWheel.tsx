import React, { useState, useRef, useEffect } from "react";

interface ColorWheelProps {
  initialColor: string;
  onColorChange: (color: string) => void;
}

export default function ColorWheel({ initialColor, onColorChange }: ColorWheelProps) {
  const [cursorPos, setCursorPos] = useState({ x: 85, y: 85 }); // Center of a 170px wheel (radius 85)
  const wheelRef = useRef<HTMLDivElement>(null);
  const wheelRectRef = useRef<DOMRect | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Initialize cursor position based on starting color if possible (or default to center)
  useEffect(() => {
    // If we have a default accent color, we can set CSS variable on mount
    document.documentElement.style.setProperty("--accent-color", initialColor);
  }, [initialColor]);

  const updateColor = (clientX: number, clientY: number) => {
    if (!wheelRef.current) return;
    const rect = wheelRectRef.current || wheelRef.current.getBoundingClientRect();
    if (!rect) return;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    let x = clientX - rect.left;
    let y = clientY - rect.top;

    const dx = x - centerX;
    const dy = y - centerY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const radius = rect.width / 2;

    if (distance > radius) {
      const angle = Math.atan2(dy, dx);
      x = centerX + radius * Math.cos(angle);
      y = centerY + radius * Math.sin(angle);
    }

    setCursorPos({ x, y });

    const angleDeg = Math.atan2(y - centerY, x - centerX) * (180 / Math.PI) + 90;
    const hue = (angleDeg + 360) % 360;
    const sat = Math.min(distance / radius, 1);
    const color = `hsl(${Math.round(hue)}, ${Math.round(sat * 100)}%, ${Math.round(100 - sat * 50)}%)`;

    // Instantly set the CSS variable on root for buttery-smooth visual updates without full component re-renders
    document.documentElement.style.setProperty("--accent-color", color);
    onColorChange(color);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    if (wheelRef.current) {
      wheelRectRef.current = wheelRef.current.getBoundingClientRect();
    }
    updateColor(e.clientX, e.clientY);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    setIsDragging(true);
    if (wheelRef.current) {
      wheelRectRef.current = wheelRef.current.getBoundingClientRect();
    }
    if (e.touches[0]) {
      updateColor(e.touches[0].clientX, e.touches[0].clientY);
    }
  };

  useEffect(() => {
    let animationFrameId: number;

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      cancelAnimationFrame(animationFrameId);
      animationFrameId = requestAnimationFrame(() => {
        updateColor(e.clientX, e.clientY);
      });
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (!isDragging) return;
      if (e.touches[0]) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = requestAnimationFrame(() => {
          updateColor(e.touches[0].clientX, e.touches[0].clientY);
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      wheelRectRef.current = null;
    };

    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove, { passive: true });
      window.addEventListener("mouseup", handleMouseUp);
      window.addEventListener("touchmove", handleTouchMove, { passive: true });
      window.addEventListener("touchend", handleMouseUp);
    }

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleMouseUp);
    };
  }, [isDragging]);

  return (
    <div className="relative w-[170px] h-[170px] mx-auto rounded-full bg-black p-[2px] shadow-[0_8px_20px_rgba(0,0,0,0.6)] overflow-hidden select-none no-double-click">
      <div
        ref={wheelRef}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
        className="w-full h-full rounded-full cursor-crosshair relative"
        style={{
          background: "conic-gradient(red, #ff0, lime, cyan, blue, #f0f, red)",
        }}
      >
        <div className="absolute inset-0 rounded-full bg-[radial-gradient(circle,rgba(255,255,255,1)_0%,rgba(255,255,255,0)_75%)] pointer-events-none" />
        <div
          className="absolute w-4 h-4 border-2 border-white rounded-full pointer-events-none shadow-[0_0_8px_rgba(0,0,0,0.6)]"
          style={{
            left: `${cursorPos.x}px`,
            top: `${cursorPos.y}px`,
            transform: "translate(-50%, -50%)",
          }}
        />
      </div>
    </div>
  );
}
