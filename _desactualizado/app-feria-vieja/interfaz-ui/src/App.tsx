import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Captions, CaptionsOff, Camera, CameraOff, Expand, Hand, Lock, PhoneOff, RotateCcw } from 'lucide-react';

type CameraState = 'idle' | 'ready' | 'blocked';

type ApiState = {
  ok?: boolean;
  translation?: string;
  source?: string;
  confidence?: number;
  timestamp?: string;
  hands?: HandForApi[];
};

type HandForApi = {
  handedness: string;
  landmarks: Array<{ x: number; y: number; z: number }>;
};

const hiddenTranslations = new Set([
  'cargando modelo',
  'lenguaje de señas',
  'lenguaje de senas',
  'leyendo seña',
  'leyendo sena',
  'modelo lsm no entrenado',
  'modelo no seguro',
  'muestra tu mano',
  'palabra aun no validada',
  'palabra aún no validada',
]);

const handConnections: Array<[number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

const predictionIntervalMs = 70;
const predictionTimeoutMs = 10000;
const captureWidth = 420;
let captureCanvas: HTMLCanvasElement | null = null;

export default function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const mirrorRef = useRef(false);
  const traceRef = useRef(true);
  const lastSendRef = useRef(0);
  const lastSeenHandRef = useRef(0);
  const sendingRef = useRef(false);
  const sessionStartedAtRef = useRef(Date.now());
  const [videoOn, setVideoOn] = useState(true);
  const [mirror, setMirror] = useState(false);
  const [showTrace, setShowTrace] = useState(true);
  const [showText, setShowText] = useState(true);
  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [translation, setTranslation] = useState('');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - sessionStartedAtRef.current) / 1000));
    }, 1000);

    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    mirrorRef.current = mirror;
  }, [mirror]);

  useEffect(() => {
    traceRef.current = showTrace;
    if (!showTrace) {
      clearCanvas(canvasRef.current);
    }
  }, [showTrace]);

  useEffect(() => {
    let alive = true;

    async function openCamera() {
      stopCamera(streamRef.current);
      streamRef.current = null;
      clearCanvas(canvasRef.current);

      if (!videoOn) {
        setCameraState('idle');
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (!alive) {
          stopCamera(stream);
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setCameraState('ready');
      } catch {
        setCameraState('blocked');
      }
    }

    openCamera();

    return () => {
      alive = false;
      stopCamera(streamRef.current);
      streamRef.current = null;
    };
  }, [videoOn]);

  useEffect(() => {
    if (!videoOn || cameraState !== 'ready') {
      clearCanvas(canvasRef.current);
      return;
    }

    let cancelled = false;
    function drawLoop() {
      if (cancelled) {
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video && canvas && video.readyState >= 2) {
        sendFramePrediction(video, canvas);
      }
      animationRef.current = window.requestAnimationFrame(drawLoop);
    }

    drawLoop();

    return () => {
      cancelled = true;
      if (animationRef.current !== null) {
        window.cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      clearCanvas(canvasRef.current);
    };
  }, [cameraState, videoOn]);

  async function sendFramePrediction(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
    const now = performance.now();
    if (sendingRef.current || now - lastSendRef.current < predictionIntervalMs) {
      return;
    }
    lastSendRef.current = now;
    sendingRef.current = true;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), predictionTimeoutMs);

    try {
      const image = captureFrame(video);
      if (!image) {
        return;
      }
      const response = await fetch('/api/prediccion-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          image,
        }),
      });
      if (!response.ok) {
        setTranslation('');
        return;
      }
      const data = (await response.json()) as ApiState;
      const hands = data.hands ?? [];
      if (traceRef.current) {
        drawHands(canvas, hands.map((hand) => hand.landmarks), mirrorRef.current, video);
      }
      if (!hands.length) {
        if (performance.now() - lastSeenHandRef.current > 1400) {
          setTranslation('');
          clearCanvas(canvas);
        }
        return;
      }
      lastSeenHandRef.current = performance.now();
      setTranslation(visibleTranslation(data.translation));
    } catch {
      setTranslation('');
    } finally {
      window.clearTimeout(timeout);
      sendingRef.current = false;
    }
  }

  function finishSession() {
    setVideoOn(false);
    setTranslation('');
    clearCanvas(canvasRef.current);
  }

  function toggleFullscreen() {
    const node = containerRef.current;
    if (!node) {
      return;
    }
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => undefined);
      return;
    }
    node.requestFullscreen().catch(() => undefined);
  }

  return (
    <div ref={containerRef} className="call-stage">
      <div className="video-layer">
        {videoOn && cameraState !== 'blocked' && (
          <video ref={videoRef} autoPlay playsInline muted className={`camera-feed ${mirror ? 'is-mirrored' : ''}`} />
        )}
        <canvas ref={canvasRef} className="hand-layer" />
      </div>

      <div className="shade-layer" />

      <header className="top-bar" aria-label="sesión del traductor">
        <button aria-label="pantalla completa" className="top-action" onClick={toggleFullscreen} type="button">
          <Expand size={18} />
        </button>

        <div className="session-title">
          <Lock size={15} />
          <span>traductor de señas</span>
        </div>

        <div className="session-meta">
          <span>{formatElapsed(elapsedSeconds)}</span>
        </div>
      </header>

      <main className="translation-zone">
        {showText && translation && <div className="translation-strip">{translation}</div>}
      </main>

      <nav className="bottom-dock" aria-label="controles del traductor">
        <DockButton active={showTrace} label="trazo" onClick={() => setShowTrace((value) => !value)}>
          <Hand size={20} />
        </DockButton>
        <DockButton active={videoOn && cameraState === 'ready'} label="cámara" onClick={() => setVideoOn((value) => !value)}>
          {videoOn ? <Camera size={20} /> : <CameraOff size={20} />}
        </DockButton>
        <DockButton active={mirror} label="espejo" onClick={() => setMirror((value) => !value)}>
          <RotateCcw size={20} />
        </DockButton>
        <DockButton active={showText} label="texto" onClick={() => setShowText((value) => !value)}>
          {showText ? <Captions size={20} /> : <CaptionsOff size={20} />}
        </DockButton>
        <DockButton active={false} end label="finalizar" onClick={finishSession}>
          <PhoneOff size={20} />
        </DockButton>
      </nav>
    </div>
  );
}

function DockButton({
  active,
  children,
  end = false,
  label,
  onClick,
}: {
  active: boolean;
  children: ReactNode;
  end?: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      aria-label={label}
      className={`dock-button ${active ? 'is-active' : ''} ${end ? 'is-end' : ''}`}
      onClick={onClick}
      type="button"
    >
      <span className="dock-icon">{children}</span>
      <span className="dock-label">{label}</span>
    </button>
  );
}

function captureFrame(video: HTMLVideoElement): string | null {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (!sourceWidth || !sourceHeight) {
    return null;
  }

  const width = Math.min(captureWidth, sourceWidth);
  const height = Math.max(1, Math.round((width * sourceHeight) / sourceWidth));
  const canvas = captureCanvas ?? document.createElement('canvas');
  captureCanvas = canvas;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) {
    return null;
  }
  context.drawImage(video, 0, 0, width, height);
  return canvas.toDataURL('image/jpeg', 0.68);
}

function drawHands(
  canvas: HTMLCanvasElement,
  hands: HandForApi['landmarks'][],
  mirror: boolean,
  video: HTMLVideoElement,
) {
  const context = canvas.getContext('2d');
  if (!context) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  context.lineCap = 'round';
  context.lineJoin = 'round';
  const videoBox = coverBox(rect.width, rect.height, video.videoWidth || rect.width, video.videoHeight || rect.height);

  for (const points of hands) {
    for (const [start, end] of handConnections) {
      const a = points[start];
      const b = points[end];
      if (!a || !b) {
        continue;
      }
      const ax = mapX(a.x, videoBox, mirror);
      const ay = videoBox.top + a.y * videoBox.height;
      const bx = mapX(b.x, videoBox, mirror);
      const by = videoBox.top + b.y * videoBox.height;
      context.strokeStyle = 'rgba(0, 0, 0, 0.55)';
      context.lineWidth = 2.8;
      context.beginPath();
      context.moveTo(ax, ay);
      context.lineTo(bx, by);
      context.stroke();
      context.strokeStyle = 'rgba(215, 232, 255, 0.96)';
      context.lineWidth = 1.45;
      context.beginPath();
      context.moveTo(ax, ay);
      context.lineTo(bx, by);
      context.stroke();
    }

    for (const point of points) {
      const x = mapX(point.x, videoBox, mirror);
      const y = videoBox.top + point.y * videoBox.height;
      context.fillStyle = 'rgba(0, 0, 0, 0.58)';
      context.beginPath();
      context.arc(x, y, 3.8, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = 'rgba(235, 245, 255, 0.98)';
      context.beginPath();
      context.arc(x, y, 2, 0, Math.PI * 2);
      context.fill();
    }
  }
}

function clearCanvas(canvas: HTMLCanvasElement | null) {
  const context = canvas?.getContext('2d');
  if (!canvas || !context) {
    return;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function coverBox(containerWidth: number, containerHeight: number, videoWidth: number, videoHeight: number) {
  const scale = Math.max(containerWidth / videoWidth, containerHeight / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  return {
    left: (containerWidth - width) / 2,
    top: (containerHeight - height) / 2,
    width,
    height,
  };
}

function mapX(value: number, box: { left: number; width: number }, mirror: boolean) {
  return box.left + (mirror ? 1 - value : value) * box.width;
}

function stopCamera(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

function visibleTranslation(value: unknown): string {
  const text = String(value ?? '').replaceAll('_', ' ').trim();
  if (!text) {
    return '';
  }
  return hiddenTranslations.has(text.toLowerCase()) ? '' : text;
}

function formatElapsed(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  return [hours, minutes, seconds].map((value) => String(value).padStart(2, '0')).join(':');
}
