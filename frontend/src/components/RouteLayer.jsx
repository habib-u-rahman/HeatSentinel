import L from "leaflet";
import { Marker, Polyline } from "react-leaflet";
import { geoJsonToLatLngs, latLonToLatLng } from "../utils/geo";

// <Polyline> is a raw Leaflet primitive -- it does NOT flip GeoJSON's
// [lon,lat] for you the way <GeoJSON> does. Every position here goes through
// geoJsonToLatLngs(); skip that once and the line renders in the ocean off
// Somalia.

const ROUTE_STYLES = {
  SHORTEST: { color: "#64748b", weight: 3, dashArray: "6 8", opacity: 0.85 },
  BALANCED: { color: "#ca8a04", weight: 2, opacity: 0.9 },
  COOLEST: { color: "#0d9488", weight: 4, opacity: 0.95 },
};
const DEFAULT_STYLE = { color: "#64748b", weight: 2, opacity: 0.85 };

function pinIcon(fillColor) {
  return L.divIcon({
    className: "",
    html:
      `<div style="width:16px;height:16px;border-radius:50% 50% 50% 0;` +
      `background:${fillColor};border:2px solid #ffffff;transform:rotate(-45deg);` +
      `box-shadow:0 2px 6px rgba(15,23,42,0.35);"></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 16],
  });
}

const START_ICON = pinIcon("#22c55e");
const END_ICON = pinIcon("#dc2626");

/**
 * @param {{lat,lon}|null} start
 * @param {{lat,lon}|null} end
 * @param {Array<{label, lambda_heat, geojson}>} routes - the Pareto family (or a single route)
 */
export default function RouteLayer({ start, end, routes }) {
  return (
    <>
      {start && <Marker position={latLonToLatLng(start)} icon={START_ICON} />}
      {end && <Marker position={latLonToLatLng(end)} icon={END_ICON} />}
      {routes?.map((route) => (
        <Polyline
          key={`${route.label ?? "route"}-${route.lambda_heat}`}
          positions={geoJsonToLatLngs(route.geojson.geometry)}
          pathOptions={ROUTE_STYLES[route.label] ?? DEFAULT_STYLE}
        />
      ))}
    </>
  );
}

export { ROUTE_STYLES };
