// GeoJSON coordinates are always [lon, lat]. Leaflet's imperative APIs
// (Polyline, Marker, Circle, ...) want [lat, lng]. react-leaflet's <GeoJSON>
// component flips correctly internally, but the moment you hand raw
// coordinates to a lower-level Leaflet primitive (a Polyline for a route
// line, a Marker for a start/end pin) you MUST flip them yourself -- get
// this backwards once and the route renders in the ocean off Somalia (0,0).
//
// This is the ONE place that flip happens. Every component imports this
// instead of writing its own [1], [0] swap.

/**
 * @param {{type: string, coordinates: any}} geometry - a GeoJSON geometry
 * @returns {Array} Leaflet-shaped coordinates: [lat, lon] for a Point,
 *   [[lat, lon], ...] for a LineString, [[[lat, lon], ...], ...] for a Polygon
 */
export function geoJsonToLatLngs(geometry) {
  if (!geometry) return [];

  switch (geometry.type) {
    case "Point": {
      const [lon, lat] = geometry.coordinates;
      return [lat, lon];
    }
    case "LineString":
      return geometry.coordinates.map(([lon, lat]) => [lat, lon]);
    case "Polygon":
      return geometry.coordinates.map((ring) => ring.map(([lon, lat]) => [lat, lon]));
    case "MultiLineString":
      return geometry.coordinates.map((line) => line.map(([lon, lat]) => [lat, lon]));
    default:
      throw new Error(`geoJsonToLatLngs: unsupported geometry type "${geometry.type}"`);
  }
}

/** A single [lon, lat] pair -> Leaflet's [lat, lon]. */
export function lonLatToLatLng([lon, lat]) {
  return [lat, lon];
}

/** {lat, lon} -> Leaflet's [lat, lon]. */
export function latLonToLatLng({ lat, lon }) {
  return [lat, lon];
}
