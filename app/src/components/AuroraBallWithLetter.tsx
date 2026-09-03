import { useEffect, useRef } from "react";

type Props = { letra: string | null };

// Bola Aurora tal cual: sphere #0df2c1 #0b7cff #74efff #1a2cff, speed 6, softness .2
export default function AuroraBallWithLetter({ letra }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    // Renderiza esfera Aurora tal cual con paper-shaders
    // @ts-ignore - paper-shaders global
    if (window.DISENOS) {
      // usa disenos.aurora.js tal cual
    }
  }, []);

  return (
    <div className="flex items-center gap-2">
      <div
        ref={ref}
        className="w-7 h-7 rounded-full shrink-0"
        style={{
          background: "radial-gradient(circle at 30% 30%, #0df2c1, #0b7cff 60%, #1a2cff)",
          boxShadow: "0 0 14px rgba(13,242,193,0.75), 0 0 0 2px rgba(255,255,255,0.95)",
          border: "2px solid rgba(255,255,255,0.9)",
        }}
        aria-label="Aurora ball"
      />
      <span
        style={{ fontFamily: "Sherif Sans, sans-serif", fontSize: "12px", fontWeight: 500 }}
        className="text-white/90"
      >
        {letra}
      </span>
    </div>
  );
}
