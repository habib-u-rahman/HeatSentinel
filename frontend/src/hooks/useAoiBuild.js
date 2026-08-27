import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";

const POLL_INTERVAL_MS = 1500;

/** Kicks off a new-city AOI build and polls its real status until it lands on
 * done/failed. Not built on useFetch -- that hook's lifecycle is "fetch once
 * when deps change", polling is a genuinely different shape (fetch, then
 * fetch again on a timer, until a terminal state), so a dedicated hook is
 * clearer than forcing one through the other.
 */
export function useAoiBuild() {
  const [status, setStatus] = useState(null); // "queued" | "running" | "done" | "failed" | null (idle)
  const [stage, setStage] = useState(null);
  const [message, setMessage] = useState(null);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const pollTimeoutRef = useRef(null);
  const jobIdRef = useRef(null);

  useEffect(() => {
    return () => {
      if (pollTimeoutRef.current) clearTimeout(pollTimeoutRef.current);
    };
  }, []);

  const poll = useCallback((jobId) => {
    api
      .aoiBuildStatus(jobId)
      .then((data) => {
        if (jobIdRef.current !== jobId) return; // a newer build superseded this one
        setStatus(data.status);
        setStage(data.stage);
        setMessage(data.message);
        setProgress(data.progress);
        if (data.status === "done") {
          setResult(data.result);
        } else if (data.status === "failed") {
          setError(data.error || "Build failed.");
        } else {
          pollTimeoutRef.current = setTimeout(() => poll(jobId), POLL_INTERVAL_MS);
        }
      })
      .catch((err) => {
        if (jobIdRef.current !== jobId) return;
        setStatus("failed");
        setError(err instanceof ApiError ? err.message : "Lost contact with the backend while building.");
      });
  }, []);

  const startBuild = useCallback(
    ({ query, bbox, radiusKm, nPoints }) => {
      setStatus("queued");
      setStage(null);
      setMessage("Starting…");
      setProgress(null);
      setResult(null);
      setError(null);

      api
        .aoiBuild({ query, bbox, radius_km: radiusKm, n_points: nPoints })
        .then((data) => {
          jobIdRef.current = data.job_id;
          setMessage(`Building ${data.city_name}…`);
          poll(data.job_id);
        })
        .catch((err) => {
          setStatus("failed");
          setError(err instanceof ApiError ? err.message : "Couldn't start the build.");
        });
    },
    [poll]
  );

  const reset = useCallback(() => {
    if (pollTimeoutRef.current) clearTimeout(pollTimeoutRef.current);
    jobIdRef.current = null;
    setStatus(null);
    setStage(null);
    setMessage(null);
    setProgress(null);
    setResult(null);
    setError(null);
  }, []);

  return { status, stage, message, progress, result, error, startBuild, reset };
}
