import InfoTooltip from "./InfoTooltip";

// Never present a literature offset as a model output -- the badge below is
// the only thing standing between "the model predicted this" and "we looked
// this number up in a paper." Keep them visually distinct.
const METHOD_BADGES = {
  model_prediction: { text: "MODEL PREDICTION", className: "border-teal-500/30 bg-teal-500/10 text-teal-400" },
  literature_offset: { text: "LITERATURE ESTIMATE", className: "border-amber-500/30 bg-amber-500/10 text-amber-400" },
};

const SCALE_MIN_C = -4;
const SCALE_MAX_C = 1;

function toScalePct(value) {
  const pct = ((value - SCALE_MIN_C) / (SCALE_MAX_C - SCALE_MIN_C)) * 100;
  return Math.max(0, Math.min(100, pct));
}

function InterventionCard({ item }) {
  const badge = METHOD_BADGES[item.method] ?? METHOD_BADGES.literature_offset;
  const [lo, hi] = item.confidence_interval;
  const left = toScalePct(lo);
  const right = toScalePct(hi);

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="font-bold capitalize text-slate-200">{item.intervention.replaceAll("_", " ")}</div>
          <span className={`mt-1 inline-block rounded border px-1.5 py-0.5 text-[8px] font-extrabold tracking-widest ${badge.className}`}>
            {badge.text}
          </span>
        </div>
        <div className={`num shrink-0 text-xl font-black ${item.value < 0 ? "text-teal-400" : "text-red-400"}`}>
          {item.value > 0 ? "+" : ""}
          {item.value.toFixed(1)}°C
        </div>
      </div>

      <div className="mt-3">
        <div className="relative h-1 w-full rounded-full bg-slate-950">
          <div
            className="absolute h-1 rounded-full bg-teal-500/70 shadow-[0_0_6px_rgba(20,184,166,0.5)]"
            style={{ left: `${left}%`, width: `${Math.max(2, right - left)}%` }}
          />
        </div>
        <div className="mt-1 flex justify-between text-[10px] text-slate-400 font-medium">
          <span>{lo.toFixed(1)}°C</span>
          <span>{hi.toFixed(1)}°C</span>
        </div>
      </div>

      {item.detail && <p className="mt-2 text-[10px] leading-relaxed text-slate-400 font-medium italic">{item.detail}</p>}
    </div>
  );
}

export default function InterventionPanel({ interventions }) {
  if (!interventions?.length) {
    return <p className="text-xs text-slate-400">No interventions available for this point.</p>;
  }
  return (
    <div className="space-y-2">
      <h3 className="label-caps flex items-center gap-1.5">
        Ranked interventions
        <InfoTooltip text="MODEL PREDICTION = estimated by a machine-learning model from this location's surface composition. LITERATURE ESTIMATE = a fixed value from published research, used when the model can't see an intervention's effect (e.g. paint reflectivity)." />
      </h3>
      {interventions.map((item) => (
        <InterventionCard key={item.intervention} item={item} />
      ))}
    </div>
  );
}
