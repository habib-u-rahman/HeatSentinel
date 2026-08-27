// Thin fetch wrapper for the HeatSentinel API. Every call takes an
// AbortSignal so callers can cancel in-flight requests (see hooks/useFetch.js)
// -- without that, rapid interactions (route clicks, slider drags) race and
// can render a stale response over a fresher one.

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function buildQuery(params = {}) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, value);
    }
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

async function request(path, { method = "GET", body, signal } = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    // network failure / backend down -- surface as a typed error, not a raw TypeError
    throw new ApiError(0, "Can't reach the HeatSentinel API. Is the backend running?");
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      if (data?.detail) detail = data.detail;
    } catch {
      // body wasn't JSON -- keep statusText
    }
    throw new ApiError(response.status, detail);
  }

  return response.json();
}

export const api = {
  health: (signal) => request("/api/health", { signal }),

  grid: (params, signal) => request(`/api/grid${buildQuery(params)}`, { signal }),
  zones: (params, signal) => request(`/api/zones${buildQuery(params)}`, { signal }),

  points: (params, signal) => request(`/api/points${buildQuery(params)}`, { signal }),
  point: (pointId, signal) => request(`/api/points/${encodeURIComponent(pointId)}`, { signal }),

  vulnerable: (params, signal) => request(`/api/vulnerable${buildQuery(params)}`, { signal }),
  alerts: (params, signal) => request(`/api/alerts${buildQuery(params)}`, { signal }),

  route: (body, signal) => request("/api/route", { method: "POST", body, signal }),

  intervention: (body, signal) => request("/api/intervention", { method: "POST", body, signal }),
  interventionsCatalog: (signal) => request("/api/interventions", { signal }),

  aoiCurrent: (signal) => request("/api/aoi/current", { signal }),
  aoiBuild: (body, signal) => request("/api/aoi/build", { method: "POST", body, signal }),
  aoiBuildStatus: (jobId, signal) => request(`/api/aoi/build/${encodeURIComponent(jobId)}`, { signal }),
};

export { BASE_URL };
