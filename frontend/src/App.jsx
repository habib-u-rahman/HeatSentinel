import { useCallback, useEffect, useMemo, useState } from "react";
import AlertsPanel from "./components/AlertsPanel";
import AoiSearch from "./components/AoiSearch";
import DataSourceBanner from "./components/DataSourceBanner";
import Legend from "./components/Legend";
import MapView from "./components/MapView";
import PoiLayer from "./components/PoiLayer";
import PointPanel from "./components/PointPanel";
import RouteLayer from "./components/RouteLayer";
import RoutePanel from "./components/RoutePanel";
import Sidebar from "./components/Sidebar";
import Toast from "./components/Toast";
import TimeScrubber from "./components/TimeScrubber";
import ZoneLayer from "./components/ZoneLayer";
import { api } from "./api/client";
import { useAlerts, useVulnerable } from "./hooks/useAlerts";
import { useZones } from "./hooks/useGrid";
import { usePoint, useSamplePoints } from "./hooks/usePoint";
import { useRoute } from "./hooks/useRoute";

const ENV_DEFAULT_BBOX = import.meta.env.VITE_AOI_BBOX || "73.03,33.58,73.07,33.61";

function BrandHeader() {
  return (
    <header className="pointer-events-auto absolute left-4 top-4 z-[1000] flex w-80 items-center justify-between rounded-xl border border-slate-800/80 bg-slate-950/80 px-4 py-2.5 shadow-xl backdrop-blur-md">
      <div className="flex items-center gap-2">
        <span className="relative flex h-2.5 w-2.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-teal-400 opacity-75"></span>
          <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-teal-500"></span>
        </span>
        <div>
          <h1 className="text-xs font-black tracking-widest text-white uppercase">HeatSentinel</h1>
          <p className="text-[9px] text-slate-400 font-semibold tracking-wider">URBAN HEAT INTEL</p>
        </div>
      </div>
      <span className="rounded bg-teal-500/10 px-1.5 py-0.5 text-[9px] font-bold text-teal-400 border border-teal-500/20">
        HACKATHON '26
      </span>
    </header>
  );
}

export default function App() {
  const [at, setAt] = useState(null); // null = server default ("now")
  // The active AOI -- starts from the env default so first paint never blocks
  // on a network call, then reconciled against GET /api/aoi/current on mount
  // (see the effect below) so a page refresh mid-demo reflects whichever
  // city a previous AoiSearch build actually swapped the backend to.
  const [aoi, setAoi] = useState({ bbox: ENV_DEFAULT_BBOX, cityName: null, degraded: false, coverageInfo: null });
  const [layerVisibility, setLayerVisibility] = useState({ zones: true, pois: true, samplePoints: false });
  const [routeStart, setRouteStart] = useState(null);
  const [routeEnd, setRouteEnd] = useState(null);
  const [lambda, setLambda] = useState(0.5);
  const [selectedPointId, setSelectedPointId] = useState(null);
  const [activeTab, setActiveTab] = useState("route");
  const [flyToTarget, setFlyToTarget] = useState(null);
  const [toastMessage, setToastMessage] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .aoiCurrent(controller.signal)
      .then((data) =>
        setAoi({
          bbox: data.aoi_bbox,
          cityName: data.city_name,
          degraded: data.degraded,
          coverageInfo:
            data.n_with_imagery != null
              ? { pct: data.mapillary_coverage_pct, nWith: data.n_with_imagery, nTotal: data.n_sample_points }
              : null,
        })
      )
      .catch(() => {}); // keep the env default -- first paint never blocks on this
    return () => controller.abort();
  }, []);

  const zones = useZones(at, aoi.bbox);
  const vulnerable = useVulnerable(at, aoi.bbox);
  const samplePoints = useSamplePoints(at, aoi.bbox);
  const routeFamily = useRoute({ start: routeStart, end: routeEnd, lambda, family: true });
  const point = usePoint(selectedPointId);
  const alerts = useAlerts({ at, aoiBbox: aoi.bbox });

  const dataSource = zones.data?.data_source ?? "fixture";

  // Surface any hook's error as a dismissable toast -- never a blank screen.
  const firstError = zones.error || vulnerable.error || alerts.error || routeFamily.error;
  useEffect(() => {
    if (firstError) setToastMessage(firstError.message);
  }, [firstError]);

  // Click-to-set-start, click-to-set-end; a third click starts a fresh pair
  // rather than appending a third point.
  const onMapClick = useCallback(
    (latlng) => {
      if (!routeStart || (routeStart && routeEnd)) {
        setRouteStart(latlng);
        setRouteEnd(null);
      } else {
        setRouteEnd(latlng);
      }
      setActiveTab("route");
    },
    [routeStart, routeEnd]
  );

  const handleSelectPoint = useCallback((pointId) => {
    setSelectedPointId(pointId);
    setActiveTab("point");
  }, []);

  const handleSelectAlert = useCallback(({ lat, lon }) => {
    setFlyToTarget([lat, lon]);
  }, []);

  const handleResetRoute = useCallback(() => {
    setRouteStart(null);
    setRouteEnd(null);
  }, []);

  // A completed AoiSearch build swaps the backend's active AOI -- old point
  // IDs and lat/lons are meaningless for a new city, so clear anything that
  // referenced the previous one rather than let them 404/point at nowhere.
  const handleAoiBuildComplete = useCallback((result) => {
    setAoi({
      bbox: result.aoi_bbox,
      cityName: result.city_name,
      degraded: result.degraded,
      coverageInfo: { pct: result.mapillary_coverage_pct, nWith: result.n_with_imagery, nTotal: result.n_sample_points },
    });
    setSelectedPointId(null);
    setRouteStart(null);
    setRouteEnd(null);
  }, []);

  const layers = useMemo(
    () => [
      { id: "zones", label: "Zones", checked: layerVisibility.zones },
      { id: "pois", label: "POIs", checked: layerVisibility.pois },
      { id: "samplePoints", label: "Photo points", checked: layerVisibility.samplePoints },
    ],
    [layerVisibility]
  );

  const handleToggleLayer = useCallback((id, checked) => {
    setLayerVisibility((prev) => ({ ...prev, [id]: checked }));
  }, []);

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-base-950 text-slate-200">
      <BrandHeader />

      <MapView aoiBbox={aoi.bbox} onMapClick={onMapClick} flyToTarget={flyToTarget}>
        <ZoneLayer data={zones.data} visible={layerVisibility.zones} />
        <PoiLayer data={vulnerable.data} kind="poi" visible={layerVisibility.pois} onSelect={handleSelectPoint} />
        <PoiLayer
          data={samplePoints.data}
          kind="sample_point"
          visible={layerVisibility.samplePoints}
          onSelect={handleSelectPoint}
        />
        {routeFamily.data && <RouteLayer start={routeStart} end={routeEnd} routes={routeFamily.data.routes} />}
      </MapView>

      <DataSourceBanner
        dataSource={dataSource}
        cityName={aoi.cityName}
        degraded={aoi.degraded}
        coveragePct={aoi.coverageInfo?.pct}
        nWithImagery={aoi.coverageInfo?.nWith}
        nSamplePoints={aoi.coverageInfo?.nTotal}
      />

      <AoiSearch onBuildComplete={handleAoiBuildComplete} />

      <Sidebar
        activeTab={activeTab}
        onChangeTab={setActiveTab}
        layers={layers}
        onToggleLayer={handleToggleLayer}
        alertCount={alerts.data?.alerts?.length ?? 0}
      >
        {activeTab === "route" && (
          <RoutePanel
            start={routeStart}
            end={routeEnd}
            lambda={lambda}
            onLambdaChange={setLambda}
            result={routeFamily.data}
            loading={routeFamily.loading}
            error={routeFamily.error}
            onReset={handleResetRoute}
          />
        )}
        {activeTab === "point" && (
          <PointPanel pointId={selectedPointId} result={point.data} loading={point.loading} error={point.error} />
        )}
        {activeTab === "alerts" && (
          <AlertsPanel result={alerts.data} loading={alerts.loading} error={alerts.error} onSelectAlert={handleSelectAlert} />
        )}
      </Sidebar>

      <Legend />
      <TimeScrubber at={at} onChange={setAt} />

      <Toast message={toastMessage} onDismiss={() => setToastMessage(null)} />
    </div>
  );
}
