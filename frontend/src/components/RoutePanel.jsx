import ErrorState from "./ErrorState";
import Skeleton from "./Skeleton";
import { riskColor } from "../utils/color";

const ROUTE_DOT_COLORS = {
  SHORTEST: "#64748b",
  BALANCED: "#ca8a04",
  COOLEST: "#0d9488",
};

export default function RoutePanel({ start, end, lambda, onLambdaChange, result, loading, error, onReset }) {
  if (!start || !end) {
    return (
      <div className="space-y-2 text-xs text-slate-400 leading-normal">
        <p className="label-caps mb-1.5 text-slate-400 font-bold tracking-widest text-[10px]">How to draw a route</p>
        <p>
          Click the map to drop a <span className="font-bold text-emerald-400">start</span> pin, then click again for a{" "}
          <span className="font-bold text-red-400">destination</span>. We'll compute the shortest path, the
          coolest path, and everything in between.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="label-caps text-slate-400 font-bold tracking-widest text-[10px]">Route comparison</h2>
        <button onClick={onReset} className="text-[10px] font-bold uppercase tracking-wider text-slate-400 transition-colors duration-150 hover:text-slate-200">
          Clear
        </button>
      </div>

      {loading && <Skeleton lines={5} />}
      {error && <ErrorState message={error.message} />}

      {result && !loading && !error && (
        <>
          {result.comparison?.same_path ? (
            <div className="rounded-lg border border-amber-500/30 bg-amber-950/20 p-3 text-xs text-amber-300 leading-normal">
              {result.comparison.summary}
            </div>
          ) : result.comparison ? (
            <ComparisonHeadline comparison={result.comparison} />
          ) : null}

          <div>
            <div className="mb-2 flex items-center justify-between">
              <span className="label-caps text-slate-400 font-bold tracking-widest text-[10px]">Heat avoidance (&lambda;)</span>
              <span className="num text-xs font-bold text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">{lambda.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={lambda}
              onChange={(e) => onLambdaChange(Number(e.target.value))}
              className="w-full accent-teal-500 cursor-pointer h-1 bg-slate-950 rounded-lg appearance-none"
            />
            <div className="flex justify-between text-[10px] text-slate-400 font-medium">
              <span>Shortest</span>
              <span>Coolest</span>
            </div>
          </div>

          <div className="space-y-1.5">
            {result.routes.map((route) => (
              <RouteRow key={route.label ?? route.lambda_heat} route={route} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function ComparisonHeadline({ comparison }) {
  const distanceLabel = `${comparison.extra_distance_m >= 0 ? "+" : ""}${Math.round(comparison.extra_distance_m)} m (${
    comparison.extra_distance_pct >= 0 ? "+" : ""
  }${comparison.extra_distance_pct.toFixed(0)}%)`;
  const wbgtLabel = `${comparison.mean_wbgt_delta_c.toFixed(1)}°C mean`;
  const doseLabel = `-${comparison.dose_reduction_pct.toFixed(0)}% exposure`;

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
        <span className="num text-md text-slate-200" title="Extra Distance">{distanceLabel}</span>
        <span className="text-slate-700">|</span>
        <span className="num text-md text-teal-400" title="Mean Temperature Difference">{wbgtLabel}</span>
        <span className="text-slate-700">|</span>
        <span className="num text-md text-emerald-400" title="Total Exposure Reduction">{doseLabel}</span>
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-slate-300 font-medium">{comparison.summary}</p>
    </div>
  );
}

function RouteRow({ route }) {
  return (
    <div className="flex items-center justify-between rounded-md bg-slate-950/40 border border-slate-900/50 px-3 py-2 text-xs">
      <div className="flex items-center gap-2">
        <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: ROUTE_DOT_COLORS[route.label] ?? "#64748b" }} />
        <span className="font-bold text-slate-200 capitalize">{route.label ? route.label.toLowerCase() : `λ = ${route.lambda_heat.toFixed(2)}`}</span>
      </div>
      <div className="flex items-center gap-3 text-slate-400">
        <span className="font-semibold">{Math.round(route.total_distance_m)} m</span>
        <span className="num" style={{ color: riskColor(route.peak_risk_band) }}>
          {route.max_wbgt_c.toFixed(1)}&deg;C
        </span>
      </div>
    </div>
  );
}
