import { useEffect } from "react";
import { MapContainer, ScaleControl, TileLayer, useMap, useMapEvents } from "react-leaflet";

// CARTO's Voyager basemap -- no API key required. Positron (light_all) was
// considered and rejected: it's too pale, and washes out the app's own
// coloured risk-band markers/zone fills/route lines; Voyager keeps enough
// visual structure (subtle land-use colour, clearer labels) to stay legible
// while still reading unambiguously as a light theme.
const DARK_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}";
const DARK_TILE_ATTRIBUTION =
  "Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012";

export function bboxToBounds(bboxStr) {
  const [minLon, minLat, maxLon, maxLat] = bboxStr.split(",").map(Number);
  return [
    [minLat, minLon],
    [maxLat, maxLon],
  ];
}

function FitBoundsOnMount({ bounds }) {
  const map = useMap();
  useEffect(() => {
    // Leaflet reads the container's size at map-creation time and caches it;
    // if the container's final CSS size (h-screen etc.) hasn't been applied
    // by the browser yet on that first tick, the map thinks the viewport is
    // whatever tiny size it started as and only ever renders tiles for that
    // area. invalidateSize() forces it to re-measure before we fit bounds.
    map.invalidateSize();
    map.fitBounds(bounds, { padding: [24, 24] });
    // Re-fit when (and only when) `bounds` itself changes -- i.e. a completed
    // AoiSearch build swapped the active AOI to a new city. Re-fitting on
    // every render would fight the user's own pan/zoom, but `bounds` is a
    // fresh array each render (bboxToBounds() recomputes it), so comparing
    // its VALUE (not identity) is what makes this fire only on a real AOI
    // change and not on every re-render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(bounds)]);
  return null;
}

function FlyToController({ target }) {
  const map = useMap();
  useEffect(() => {
    if (target) {
      map.flyTo(target, Math.max(map.getZoom(), 15), { duration: 0.8 });
    }
  }, [target, map]);
  return null;
}

function ClickHandler({ onMapClick }) {
  useMapEvents({
    click(e) {
      onMapClick?.({ lat: e.latlng.lat, lon: e.latlng.lng });
    },
  });
  return null;
}

/**
 * The map. preferCanvas is load-bearing: with thousands of grid/zone/POI
 * features, the default SVG renderer freezes the browser -- canvas doesn't.
 */
export default function MapView({ aoiBbox, onMapClick, flyToTarget, children }) {
  const bounds = bboxToBounds(aoiBbox);

  return (
    <MapContainer
      className="absolute inset-0 z-0"
      bounds={bounds}
      preferCanvas
      zoomControl={false}
      style={{ height: "100%", width: "100%", background: "#090d16" }}
    >
      <TileLayer url={DARK_TILE_URL} attribution={DARK_TILE_ATTRIBUTION} maxZoom={20} />
      <FitBoundsOnMount bounds={bounds} />
      <FlyToController target={flyToTarget} />
      <ClickHandler onMapClick={onMapClick} />
      <ScaleControl position="bottomleft" />
      {children}
    </MapContainer>
  );
}
