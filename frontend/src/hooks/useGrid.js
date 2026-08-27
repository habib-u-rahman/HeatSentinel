import { api } from "../api/client";
import { useFetch } from "./useFetch";

// Fetched on mount and on explicit timestamp change ONLY -- never on map
// pan/zoom. `at` is either null (server default: now) or an ISO string picked
// via TimeScrubber, so this never fires on every map move. `aoiBbox` is a
// second explicit trigger of the same kind: it's never sent to the backend
// (which already serves whatever AOI is active in app.state), it's here
// purely so a completed AoiSearch build -- which swaps the active AOI --
// forces a refetch, the same way picking a new timestamp does.

export function useGrid(at, aoiBbox) {
  return useFetch((signal) => api.grid(at ? { at } : {}, signal), [at, aoiBbox]);
}

export function useZones(at, aoiBbox) {
  return useFetch((signal) => api.zones(at ? { at } : {}, signal), [at, aoiBbox]);
}
