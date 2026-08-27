import InfoTooltip from "./InfoTooltip";
import { RISK_COLORS, THERMAL_SCALE_STOPS } from "../utils/color";

export default function Legend() {
  return (
    <div className="pointer-events-auto absolute bottom-6 left-4 z-[1000] rounded-xl border border-slate-800/80 bg-slate-900/75 p-3.5 shadow-2xl backdrop-blur-md">
      <div className="label-caps mb-2 text-slate-400 font-bold tracking-widest text-[10px]">Risk band</div>
      <div className="flex items-center gap-3">
        {Object.entries(RISK_COLORS).map(([band, color]) => (
          <div key={band} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: color }} />
            <span className="text-[10px] text-slate-300 font-semibold capitalize">{band.toLowerCase()}</span>
          </div>
        ))}
      </div>

      <div className="label-caps mb-1.5 mt-3.5 flex items-center gap-1.5 text-slate-400 font-bold tracking-widest text-[10px]">
        WBGT scale
        <InfoTooltip text="Wet-Bulb Globe Temperature: a heat-stress index computed from the synthetic temperature grid plus real humidity." />
      </div>
      <div
        className="h-2 w-40 rounded-full shadow-inner shadow-black/40"
        style={{ background: `linear-gradient(to right, ${THERMAL_SCALE_STOPS.map((s) => s.color).join(", ")})` }}
      />
      <div className="mt-1 flex justify-between text-[10px] text-slate-400 font-medium">
        <span>cool</span>
        <span>hot</span>
      </div>
    </div>
  );
}
