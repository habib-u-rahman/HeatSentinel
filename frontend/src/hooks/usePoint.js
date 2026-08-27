import { api } from "../api/client";
import { useFetch } from "./useFetch";

/** The full detail for one sample point or POI: photo, overlay, surface
 * composition, current WBGT, ranked interventions. This is what drives
 * PointPanel. */
export function usePoint(pointId) {
  return useFetch((signal) => api.point(pointId, signal), [pointId], { enabled: Boolean(pointId) });
}

/** The clickable map layer of sample points (id + risk_band only) -- fetched
 * on mount and on explicit timestamp change, same as grid/zones. `aoiBbox` is
 * an explicit-trigger-only dependency (see useGrid.js) so a completed AOI
 * build refetches this layer for the new city. */
export function useSamplePoints(at, aoiBbox) {
  return useFetch((signal) => api.points(at ? { at } : {}, signal), [at, aoiBbox]);
}
