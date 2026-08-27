export default function ErrorState({ message, onRetry }) {
  return (
    <div className="rounded-lg border border-risk-critical/40 bg-risk-critical/10 p-3 text-sm">
      <div className="font-semibold text-red-800">Couldn't load this</div>
      <div className="mt-1 text-red-700/90">{message || "Unknown error."}</div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 rounded-md bg-red-500/20 px-2 py-1 text-xs font-semibold text-red-800 transition-colors duration-150 hover:bg-red-500/30"
        >
          Retry
        </button>
      )}
    </div>
  );
}
