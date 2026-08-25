export default function Skeleton({ lines = 3, className = "" }) {
  return (
    <div className={`animate-pulse space-y-2 ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="h-3 rounded bg-base-700/70" style={{ width: `${100 - i * 14}%` }} />
      ))}
    </div>
  );
}
