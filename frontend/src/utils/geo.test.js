import { describe, expect, it } from "vitest";
import { geoJsonToLatLngs, latLonToLatLng, lonLatToLatLng } from "./geo";

describe("geoJsonToLatLngs", () => {
  it("flips a Point from [lon, lat] to [lat, lon]", () => {
    expect(geoJsonToLatLngs({ type: "Point", coordinates: [73.05, 33.6] })).toEqual([33.6, 73.05]);
  });

  it("flips every vertex of a LineString", () => {
    const result = geoJsonToLatLngs({
      type: "LineString",
      coordinates: [
        [73.0, 33.5],
        [73.1, 33.6],
      ],
    });
    expect(result).toEqual([
      [33.5, 73.0],
      [33.6, 73.1],
    ]);
  });

  it("flips every ring of a Polygon", () => {
    const result = geoJsonToLatLngs({
      type: "Polygon",
      coordinates: [
        [
          [73.0, 33.5],
          [73.1, 33.5],
          [73.1, 33.6],
          [73.0, 33.6],
          [73.0, 33.5],
        ],
      ],
    });
    expect(result[0]).toEqual([
      [33.5, 73.0],
      [33.5, 73.1],
      [33.6, 73.1],
      [33.6, 73.0],
      [33.5, 73.0],
    ]);
  });

  it("flips every line of a MultiLineString", () => {
    const result = geoJsonToLatLngs({
      type: "MultiLineString",
      coordinates: [
        [
          [73.0, 33.5],
          [73.1, 33.6],
        ],
        [
          [73.2, 33.7],
          [73.3, 33.8],
        ],
      ],
    });
    expect(result).toEqual([
      [
        [33.5, 73.0],
        [33.6, 73.1],
      ],
      [
        [33.7, 73.2],
        [33.8, 73.3],
      ],
    ]);
  });

  it("returns an empty array for null/undefined geometry rather than throwing", () => {
    expect(geoJsonToLatLngs(null)).toEqual([]);
    expect(geoJsonToLatLngs(undefined)).toEqual([]);
  });

  it("throws on an unsupported geometry type instead of silently misrendering", () => {
    expect(() => geoJsonToLatLngs({ type: "MultiPoint", coordinates: [] })).toThrow(/unsupported geometry type/i);
  });

  it("never produces a coordinate that would render off Somalia (0,0) for real AOI data", () => {
    // this is the exact regression the whole module exists to prevent
    const flipped = geoJsonToLatLngs({ type: "Point", coordinates: [73.05, 33.6] });
    expect(flipped).not.toEqual([0, 0]);
    expect(flipped[0]).toBeCloseTo(33.6); // lat
    expect(flipped[1]).toBeCloseTo(73.05); // lon
  });
});

describe("lonLatToLatLng", () => {
  it("flips a raw [lon, lat] pair", () => {
    expect(lonLatToLatLng([73.05, 33.6])).toEqual([33.6, 73.05]);
  });
});

describe("latLonToLatLng", () => {
  it("converts a {lat, lon} object to a Leaflet [lat, lon] pair", () => {
    expect(latLonToLatLng({ lat: 33.6, lon: 73.05 })).toEqual([33.6, 73.05]);
  });
});
