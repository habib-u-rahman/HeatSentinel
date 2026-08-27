import { useEffect, useRef, useState } from "react";

const DEBOUNCE_MS = 300; // dragging the slider must not fire a fetch per frame

function isoForHour(hour) {
  const now = new Date();
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), hour, 0, 0)).toISOString();
}

/** Hour-of-day scrubber. The slider thumb moves immediately (local state);
 * the actual grid/zones refetch (via onChange) is debounced 300ms after the
 * user stops moving it -- this IS the "explicit timestamp change" gotcha #3
 * allows, but a raw drag still fires many intermediate values without it. */
export default function TimeScrubber({ at, onChange }) {
  const [hour, setHour] = useState(() => (at ? new Date(at).getUTCHours() : new Date().getUTCHours()));
  const timeoutRef = useRef(null);

  useEffect(() => () => clearTimeout(timeoutRef.current), []);

  const handleSlide = (h) => {
    setHour(h);
    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => onChange(isoForHour(h)), DEBOUNCE_MS);
  };

  const handleNow = () => {
    clearTimeout(timeoutRef.current);
    setHour(new Date().getUTCHours());
    onChange(null);
  };

  return (
    <div className="pointer-events-auto absolute bottom-6 left-1/2 z-[1000] w-[26rem] -translate-x-1/2 rounded-xl border border-slate-800/80 bg-slate-900/75 px-4 py-3.5 shadow-2xl backdrop-blur-md transition-all duration-300">
      <div className="mb-2 flex items-center justify-between">
        <span className="label-caps text-slate-400 font-bold tracking-widest text-[10px]">Time of day</span>
        <div className="flex items-center gap-2">
          <span className="num text-xs font-bold text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">{String(hour).padStart(2, "0")}:00 UTC</span>
          {at && (
            <button onClick={handleNow} className="text-[10px] font-bold uppercase tracking-widest text-teal-400 hover:text-teal-300 transition-colors">
              Now
            </button>
          )}
        </div>
      </div>
      <input
        type="range"
        min={0}
        max={23}
        step={1}
        value={hour}
        onChange={(e) => handleSlide(Number(e.target.value))}
        className="w-full accent-teal-500 cursor-pointer h-1 bg-slate-950 rounded-lg appearance-none"
      />
      <div className="mt-1 flex justify-between text-[10px] text-slate-400 font-medium">
        <span>00:00</span>
        <span>12:00</span>
        <span>23:00</span>
      </div>
    </div>
  );
}
