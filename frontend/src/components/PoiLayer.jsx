import L from "leaflet";
import { GeoJSON } from "react-leaflet";
import { riskColor } from "../utils/color";

/**
 * Renders one GeoJSON Point FeatureCollection as clickable circle markers.
 * Used for BOTH vulnerable-population-proxy POIs (schools, hospitals, bus
 * stops, ...) and the sample-point layer (the 317 locations with a real
 * street photo) -- same interaction, different visual weight, both feeding
 * PointPanel via onSelect(id). CircleMarker draws on the canvas renderer
 * (preferCanvas on MapContainer), which is what keeps hundreds of markers
 * smooth.
 */
export default function PoiLayer({ data, kind = "poi", visible = true, onSelect }) {
  if (!visible || !data?.features?.length) return null;

  const isSamplePoint = kind === "sample_point";

  const pointToLayer = (feature, latlng) => {
    const band = feature.properties.risk_band;
    return L.circleMarker(latlng, {
      radius: isSamplePoint ? 5 : 6,
      weight: isSamplePoint ? 1.5 : 1,
      color: "#1e293b",
      fillColor: riskColor(band),
      fillOpacity: isSamplePoint ? 0.85 : 0.9,
    });
  };

  const onEachFeature = (feature, layer) => {
    const props = feature.properties;
    const id = isSamplePoint ? props.point_id : props.poi_id;
    const label = isSamplePoint
      ? `Street photo point &middot; ${props.risk_band ?? "unknown"}`
      : `${props.name || props.category}${props.risk_band ? ` &middot; ${props.risk_band}` : ""}`;
    layer.bindTooltip(label, { sticky: true, className: "heatsentinel-tooltip" });
    layer.on("click", (e) => {
      // Leaflet does NOT stop a layer's click from also reaching the map's
      // own click handler (true for both SVG and canvas rendering) -- without
      // this, clicking a marker ALSO registers as a route-drawing map click.
      L.DomEvent.stopPropagation(e);
      onSelect?.(id);
    });
  };

  return (
    <GeoJSON
      key={`${kind}:${data.observed_at}`}
      data={data}
      pointToLayer={pointToLayer}
      onEachFeature={onEachFeature}
    />
  );
}
