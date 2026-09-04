import { Play, Loader2 } from "lucide-react";

type Props = {
  onTrain: () => void;
  training: boolean;
  count: number;
};

export default function TrainButton({ onTrain, training, count }: Props) {
  return (
    <button
      onClick={onTrain}
      disabled={training || count === 0}
      className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#0a0a0a] text-white border border-white/10 shadow-[0_8px_30px_rgba(0,0,0,0.6)] hover:bg-white hover:text-black transition disabled:opacity-40 disabled:cursor-not-allowed"
    >
      {training ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
      <span className="text-sm font-medium">
        {training ? "Entrenando..." : `Entrenar (${count} fotos)`}
      </span>
    </button>
  );
}
