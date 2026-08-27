const TABS = [
  { id: "route", label: "Route" },
  { id: "point", label: "Point" },
  { id: "alerts", label: "Alerts" },
];

/** A floating translucent panel over the map -- deliberately not a fixed
 * column, so the map stays the hero and this reads as chrome on top of it. */
export default function Sidebar({
  activeTab,
  onChangeTab,
  layers,
  onToggleLayer,
  alertCount,
  children,
}) {
  return (
    <aside className="pointer-events-auto absolute right-4 top-20 z-[1000] flex max-h-[calc(100vh-7rem)] w-[26rem] flex-col overflow-hidden rounded-2xl border border-slate-800/80 bg-slate-900/75 shadow-2xl backdrop-blur-md transition-all duration-300">
      <div className="flex items-center gap-1.5 border-b border-slate-800/80 p-2">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onChangeTab(tab.id)}
            className={`relative flex-1 rounded-lg px-3 py-1.5 text-xs font-bold uppercase tracking-wider transition-all duration-150 ${
              activeTab === tab.id
                ? "bg-teal-600/20 text-teal-400 border border-teal-500/30 shadow-[0_0_8px_rgba(20,184,166,0.15)]"
                : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border border-transparent"
            }`}
          >
            {tab.label}
            {tab.id === "alerts" && alertCount > 0 && (
              <span className="absolute -right-1 -top-1 flex h-4 min-w-4 items-center justify-center rounded-full bg-risk-critical px-1 text-[9px] font-bold text-white shadow-[0_0_6px_rgba(220,38,38,0.5)] animate-pulse">
                {alertCount > 99 ? "99+" : alertCount}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="thin-scroll flex-1 overflow-y-auto p-4 text-slate-300">{children}</div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 border-t border-slate-800/80 px-4 py-3">
        {layers.map((layer) => (
          <label key={layer.id} className="flex cursor-pointer items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors">
            <input
              type="checkbox"
              checked={layer.checked}
              onChange={(e) => onToggleLayer(layer.id, e.target.checked)}
              className="h-3.5 w-3.5 rounded border-slate-800 bg-slate-950 text-teal-500 accent-teal-500 focus:ring-0 focus:ring-offset-0"
            />
            <span className="font-semibold">{layer.label}</span>
          </label>
        ))}
      </div>
    </aside>
  );
}
