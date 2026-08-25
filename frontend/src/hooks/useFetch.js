import { useEffect, useState } from "react";

/**
 * Shared fetch lifecycle for every hook in this app: AbortController on every
 * request (cancelled in the effect cleanup) so rapid interactions -- route
 * clicks, slider drags -- can't race and render a stale response over a
 * fresher one; optional debounce for continuous inputs like the lambda slider.
 *
 * @param {(signal: AbortSignal) => Promise<any>} fetcher
 * @param {any[]} deps
 * @param {{enabled?: boolean, debounceMs?: number}} [options]
 */
export function useFetch(fetcher, deps, { enabled = true, debounceMs = 0 } = {}) {
  const [state, setState] = useState({ data: null, loading: enabled, error: null });

  useEffect(() => {
    if (!enabled) {
      setState({ data: null, loading: false, error: null });
      return undefined;
    }

    const controller = new AbortController();
    let timeoutId;
    setState((prev) => ({ ...prev, loading: true, error: null }));

    const run = () => {
      fetcher(controller.signal)
        .then((data) => setState({ data, loading: false, error: null }))
        .catch((error) => {
          if (error.name === "AbortError") return; // cancelled, not a real failure
          setState({ data: null, loading: false, error });
        });
    };

    if (debounceMs > 0) {
      timeoutId = setTimeout(run, debounceMs);
    } else {
      run();
    }

    return () => {
      controller.abort();
      if (timeoutId) clearTimeout(timeoutId);
    };
    // deps is caller-controlled and intentionally spread below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
