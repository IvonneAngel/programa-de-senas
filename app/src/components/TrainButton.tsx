type TrainButtonProps = {
  onTrain: () => void;
  training: boolean;
  count: number;
};

export default function TrainButton({ onTrain, training, count }: TrainButtonProps) {
  const disabled = training || count === 0;

  return (
    <button
      type="button"
      onClick={onTrain}
      disabled={disabled}
      title={count === 0 ? "Captura al menos un frame antes de entrenar" : "Solicitar entrenamiento al backend"}
      className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-emerald-400 text-black border border-black/10 shadow-[0_8px_30px_rgba(0,0,0,0.6)] hover:bg-emerald-300 transition disabled:opacity-40 disabled:cursor-not-allowed"
    >
      <span className="text-sm font-medium">
        {training ? "Entrenando…" : `Entrenar (${count})`}
      </span>
    </button>
  );
}
