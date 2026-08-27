import { useState } from "react";
import { useAoiBuild } from "../hooks/useAoiBuild";
import ErrorState from "./ErrorState";

const DEFAULT_N_POINTS = 60;
const DEFAULT_RADIUS_KM = 1.0;

const STAGE_LABELS = {
  graph: "Street network",
  sampling: "Sample locations",
  imagery: "Street photos + computer vision",
  pois: "Points of interest",
};

/** The location picker. Building a new city is a real, multi-minute pipeline
 * (a live OSM/Overpass fetch, real Mapillary photo downloads, real CV
 * inference per photo) -- every number shown here comes straight from the
 * polled job status, never a client-side animation, so what's on screen is
 * always literally true at that moment. */
export default function AoiSearch({ onBuildComplete }) {
  const [query, setQuery] = useState("");
  const { status, stage, message, progress, result, error, startBuild, reset } = useAoiBuild();

  const busy = status === "queued" || status === "running";

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim() || busy) return;
    startBuild({ query: query.trim(), radiusKm: DEFAULT_RADIUS_KM, nPoints: DEFAULT_N_POINTS });
  };

  const handleUseResult = () => {
    if (result) onBuildComplete?.(result);
    reset();
    setQuery("");
  };

  return (
    <div className="pointer-events-auto absolute left-4 top-20 z-[1000] w-80 rounded-xl border border-slate-800/80 bg-slate-900/75 p-3 shadow-2xl backdrop-blur-md transition-all duration-300">
      <div className="label-caps mb-2 text-slate-400 font-bold tracking-widest text-[10px]">Build a new city</div>

      <form onSubmit={handleSubmit} className="flex gap-1.5">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={busy}
          placeholder="e.g. Lahore, Pakistan"
          className="w-full rounded-md border border-slate-800 bg-slate-950/70 px-2.5 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 focus:border-teal-500 focus:ring-1 focus:ring-teal-500/20 focus:outline-none disabled:opacity-50 transition-all"
        />
        <button
          type="submit"
          disabled={busy || !query.trim()}
          className="shrink-0 rounded-md bg-teal-600 px-3.5 py-1.5 text-xs font-bold uppercase tracking-wider text-white transition-all duration-150 hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-40 shadow-md shadow-teal-900/25"
        >
          {busy ? "Building…" : "Build"}
        </button>
      </form>

      {busy && (
        <div className="mt-2.5 space-y-1">
          <div className="flex items-center justify-between text-[10px] text-slate-400">
            <span>{stage ? STAGE_LABELS[stage] || stage : "Starting"}</span>
            {progress && progress.total > 0 && (
              <span className="num text-teal-400">
                {progress.current}/{progress.total}
              </span>
            )}
          </div>
          {progress && progress.total > 0 && (
            <div className="h-1 w-full overflow-hidden rounded-full bg-slate-950">
              <div
                className="h-full rounded-full bg-gradient-to-r from-teal-500 to-cyan-400 transition-[width] duration-200 shadow-[0_0_8px_rgba(20,184,166,0.5)]"
                style={{ width: `${Math.min(100, (progress.current / progress.total) * 100)}%` }}
              />
            </div>
          )}
          <div className="text-[10px] text-slate-400 italic">{message}</div>
          <div className="text-[9px] text-slate-500">
            Fetching OpenStreetMap, downloading street photos and executing AI inference.
          </div>
        </div>
      )}

      {status === "failed" && <div className="mt-2.5"><ErrorState message={error} onRetry={reset} /></div>}

      {status === "done" && result && (
        <div className="mt-2.5 rounded-lg border border-teal-500/30 bg-teal-950/20 p-2.5 text-xs">
          <div className="font-bold text-teal-400">{result.city_name} is ready</div>
          <div className="mt-1 text-[11px] text-slate-300 leading-normal">
            {result.n_with_imagery}/{result.n_sample_points} sample points mapped with street imagery (
            {result.mapillary_coverage_pct.toFixed(0)}%), and {result.n_pois} points of interest detected.
          </div>
          {result.degraded && (
            <div className="mt-1 text-[10px] text-amber-400 leading-normal">
              Note: Coverage is sparse. The model representation might be thin, but all points are active.
            </div>
          )}
          <button
            onClick={handleUseResult}
            className="mt-2.5 w-full rounded bg-teal-500/20 py-1 text-center text-xs font-bold text-teal-300 border border-teal-500/30 transition-all hover:bg-teal-500/30"
          >
            Switch map viewport to {result.city_name}
          </button>
        </div>
      )}
    </div>
  );
}
