import { useState } from "react";
import { BASE_URL } from "../api/client";
import ErrorState from "./ErrorState";
import InfoTooltip from "./InfoTooltip";
import InterventionPanel from "./InterventionPanel";
import Skeleton from "./Skeleton";
import { riskColor } from "../utils/color";

const BUCKET_COLORS = {
  road: "#475569",
  sidewalk: "#94a3b8",
  built: "#78716c",
  vegetation: "#22c55e",
  sky: "#0891b2",
  other: "#a855f7",
};
const BUCKET_ORDER = ["road", "sidewalk", "built", "vegetation", "sky", "other"];

function PhotoCompare({ imageUrl, overlayUrl }) {
  const [opacity, setOpacity] = useState(0.5);

  if (!imageUrl) {
    return (
      <div className="flex h-40 items-center justify-center rounded-lg border border-slate-800 bg-slate-950/40 text-xs text-slate-400">
        No photo on file for this point
      </div>
    );
  }

  return (
    <div>
      <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-black shadow-lg">
        <img src={`${BASE_URL}${imageUrl}`} alt="Street view" className="block w-full" loading="lazy" />
        {overlayUrl && (
          <img
            src={`${BASE_URL}${overlayUrl}`}
            alt="Segmentation overlay"
            className="absolute inset-0 h-full w-full"
            style={{ opacity }}
            loading="lazy"
          />
        )}
      </div>
      {overlayUrl && (
        <div className="mt-1.5">
          <div className="flex justify-between text-[10px] text-slate-400 font-medium">
            <span>Photo</span>
            <span>Segmentation</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="w-full accent-teal-500 cursor-pointer h-1 bg-slate-950 rounded-lg appearance-none"
          />
        </div>
      )}
    </div>
  );
}

function SurfaceBar({ buckets }) {
  return (
    <div>
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-950 shadow-inner">
        {BUCKET_ORDER.map((key) => (
          <div
            key={key}
            style={{ width: `${buckets[key] * 100}%`, background: BUCKET_COLORS[key] }}
            title={`${key}: ${(buckets[key] * 100).toFixed(0)}%`}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1.5 text-[10px] text-slate-400 font-semibold">
        {BUCKET_ORDER.map((key) => (
          <span key={key} className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: BUCKET_COLORS[key] }} />
            <span className="capitalize">{key}</span> {(buckets[key] * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  );
}

/** THE DEMO CENTREPIECE: real street photo + segmentation overlay, surface
 * composition, current heat readout, and ranked interventions for one point. */
export default function PointPanel({ pointId, result, loading, error }) {
  if (!pointId) {
    return (
      <p className="text-xs text-slate-400 leading-normal">
        Click a marker — a school, clinic, bus stop, or street-photo point — to view its heat profile.
      </p>
    );
  }
  if (loading) return <Skeleton lines={7} />;
  if (error) {
    return <ErrorState message={error.status === 404 ? "No street-level data for this location." : error.message} />;
  }
  if (!result) return null;

  const color = riskColor(result.risk_band);

  return (
    <div className="space-y-4">
      <div>
        <div className="label-caps text-slate-400 font-bold tracking-widest text-[10px]">{result.point_id}</div>
        <div className="mt-1 flex items-baseline gap-3">
          <span className="num text-4xl drop-shadow-[0_2px_10px_rgba(0,0,0,0.3)]" style={{ color }}>
            {result.wbgt_c.toFixed(1)}°C
          </span>
          <span
            className="rounded px-1.5 py-0.5 text-[9px] font-extrabold uppercase tracking-widest"
            style={{ background: `${color}25`, color, border: `1px solid ${color}35` }}
          >
            {result.risk_band}
          </span>
        </div>
        <div className="text-[10px] text-slate-400 font-medium mt-0.5">air temperature {result.temp_c.toFixed(1)}°C &middot; WBGT index shown</div>
      </div>

      <PhotoCompare imageUrl={result.image_url} overlayUrl={result.overlay_url} />

      <div>
        <h3 className="label-caps mb-2 flex items-center gap-1.5 text-slate-400 font-bold tracking-widest text-[10px]">
          Surface composition
          <InfoTooltip text="Estimated from a computer-vision model (semantic segmentation) run on the real street photo above." />
        </h3>
        <SurfaceBar buckets={result.buckets} />
      </div>

      <InterventionPanel interventions={result.interventions} />
    </div>
  );
}
