/** Deliberate, persistent disclosure -- never hide this while the temperature
 * layer is synthetic. This is how the demo stays honest on stage. A second,
 * separate banner appears when the active AOI's street imagery coverage is
 * real but sparse (see app/api/routes/aoi.py's degraded threshold) -- shown,
 * never hidden, same honesty principle applied to a different kind of gap. */
export default function DataSourceBanner({ dataSource, cityName, degraded, coveragePct, nWithImagery, nSamplePoints }) {
  const showFixtureBanner = dataSource === "fixture";
  if (!showFixtureBanner && !degraded) return null;

  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-[1100] flex flex-col items-center gap-1.5 px-4 pt-3.5">
      {showFixtureBanner && (
        <div className="pointer-events-auto flex max-w-2xl items-center gap-2.5 rounded-full border border-amber-500/25 bg-amber-500/10 px-4 py-1.5 text-xs font-bold text-amber-200 shadow-xl backdrop-blur-md">
          <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.6)]" />
          <span>
            Temperature Layer: <span className="font-extrabold text-amber-300">Synthetic</span> (FortyGuard API Pending). Imagery, street network, and CV segmentations are real {cityName || "Rawalpindi"} data.
          </span>
        </div>
      )}
      {degraded && (
        <div className="pointer-events-auto flex max-w-2xl items-center gap-2.5 rounded-full border border-orange-500/25 bg-orange-500/10 px-4 py-1.5 text-xs font-bold text-orange-200 shadow-xl backdrop-blur-md">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-orange-400" />
          <span>
            Sparse Street Imagery for {cityName}: only {nWithImagery}/{nSamplePoints} sample points (
            {coveragePct != null ? coveragePct.toFixed(0) : "?"}%) had usable Mapillary coverage.
          </span>
        </div>
      )}
    </div>
  );
}
