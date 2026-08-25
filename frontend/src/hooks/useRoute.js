import { api } from "../api/client";
import { useFetch } from "./useFetch";

const DEBOUNCE_MS = 300;

/**
 * The Pareto route family (or a single route) between start and end.
 * Debounced 300ms so dragging the lambda slider doesn't fire a request per
 * frame; disabled entirely until both start and end are set.
 *
 * @param {{start: {lat,lon}|null, end: {lat,lon}|null, lambda?: number, family?: boolean}} params
 */
export function useRoute({ start, end, lambda = 0.5, family = false }) {
  const enabled = Boolean(start && end);
  return useFetch(
    (signal) => api.route({ start, end, lambda_heat: lambda, family }, signal),
    [start?.lat, start?.lon, end?.lat, end?.lon, lambda, family],
    { enabled, debounceMs: DEBOUNCE_MS }
  );
}
