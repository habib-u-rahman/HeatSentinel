import { useEffect } from "react";

/** A red toast for backend-down / request failures -- never a blank white
 * screen when something goes wrong. */
export default function Toast({ message, onDismiss, durationMs = 6000 }) {
  useEffect(() => {
    if (!message) return undefined;
    const timer = setTimeout(onDismiss, durationMs);
    return () => clearTimeout(timer);
  }, [message, onDismiss, durationMs]);

  if (!message) return null;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-4 z-[2000] flex justify-center">
      <div className="pointer-events-auto flex items-center gap-3 rounded-lg border border-red-500/40 bg-red-950/95 px-4 py-2.5 text-sm text-red-100 shadow-2xl backdrop-blur-sm">
        <span className="h-2 w-2 shrink-0 rounded-full bg-red-400" />
        <span>{message}</span>
        <button onClick={onDismiss} className="ml-2 text-lg leading-none text-red-300 hover:text-white">
          &times;
        </button>
      </div>
    </div>
  );
}
