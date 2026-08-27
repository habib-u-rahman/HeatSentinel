import { api } from "../api/client";
import { useFetch } from "./useFetch";

export function useAlerts({ at, minSeverity, aoiBbox } = {}) {
  return useFetch((signal) => api.alerts({ at, min_severity: minSeverity }, signal), [at, minSeverity, aoiBbox]);
}

/** Vulnerable-population-proxy POIs (schools, hospitals, bus stops, ...) for
 * PoiLayer -- closely related to alerts (poi_at_risk alerts reference these
 * same POIs), fetched on mount / explicit timestamp change only. `aoiBbox` is
 * an explicit-trigger-only dependency (see useGrid.js) so a completed AOI
 * build refetches this layer for the new city. */
export function useVulnerable(at, aoiBbox) {
  return useFetch((signal) => api.vulnerable(at ? { at } : {}, signal), [at, aoiBbox]);
}
