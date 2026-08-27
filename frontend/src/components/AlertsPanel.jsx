import { useState } from "react";
import ErrorState from "./ErrorState";
import Skeleton from "./Skeleton";
import { riskColor } from "../utils/color";

const CATEGORY_LABELS = {
  zone_critical: "Zone critical",
  poi_at_risk: "POI at risk",
  rapid_rise: "Rapid rise",
};

function AlertRow({ alert, onClick }) {
  const color = riskColor(alert.severity);
  return (
    <button
      onClick={onClick}
      className="flex w-full items-start gap-2.5 rounded-md border-l-2 bg-base-800/50 px-3 py-2 text-left transition-colors duration-150 hover:bg-base-800"
      style={{ borderColor: color }}
    >
      <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[10px] font-bold uppercase tracking-wide text-slate-500">
            {CATEGORY_LABELS[alert.category] ?? alert.category}
          </span>
          <span className="num text-xs" style={{ color }}>
            {alert.wbgt_c.toFixed(1)}°C
          </span>
        </div>
        <p className="mt-0.5 text-xs leading-snug text-slate-700">{alert.message}</p>
      </div>
    </button>
  );
}

/** Collapsible, severity-coloured list from /api/alerts. Clicking an alert
 * pans the map to it via onSelectAlert({lat, lon}). */
export default function AlertsPanel({ result, loading, error, onSelectAlert }) {
  const [collapsed, setCollapsed] = useState(false);

  if (loading) return <Skeleton lines={6} />;
  if (error) return <ErrorState message={error.message} />;

  const alerts = result?.alerts ?? [];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="label-caps">Active alerts ({alerts.length})</h2>
        <button onClick={() => setCollapsed((c) => !c)} className="text-xs text-slate-600 hover:text-slate-900">
          {collapsed ? "Expand" : "Collapse"}
        </button>
      </div>

      <p className="text-[11px] leading-relaxed text-slate-500">
        Alerts blend zone temperature data with an OSM points-of-interest PROXY for vulnerable populations — not real
        population or census data.
      </p>

      {!collapsed && (
        <div className="space-y-1.5">
          {alerts.length === 0 && <p className="text-sm text-slate-500">No active alerts right now.</p>}
          {alerts.map((alert) => (
            <AlertRow key={alert.alert_id} alert={alert} onClick={() => onSelectAlert?.({ lat: alert.lat, lon: alert.lon })} />
          ))}
        </div>
      )}
    </div>
  );
}
