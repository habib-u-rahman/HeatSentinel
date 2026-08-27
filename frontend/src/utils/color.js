// Risk bands: fixed colours per the design spec, varying LIGHTNESS as well as
// hue (not just hue) so the sequence survives colour-blindness and a washed-out
// projector: green (light/mid) -> yellow (mid) -> orange (mid-dark) -> red (dark).
// Kept in sync with tailwind.config.js's risk.* tokens (this file can't import
// the Tailwind config directly, since it's also used for map marker/fill
// colours that Tailwind classes can't reach) -- slightly darker than the pure
// green/yellow/orange would be, for contrast against the light theme's
// near-white background.
export const RISK_COLORS = {
  SAFE: "#16a34a",
  CAUTION: "#ca8a04",
  DANGER: "#ea580c",
  CRITICAL: "#dc2626",
};

export function riskColor(band) {
  return RISK_COLORS[band] ?? "#64748b"; // slate-500 fallback for unknown/null bands
}

// Continuous thermal scale for raw temp/WBGT values: deep blue -> teal -> amber
// -> red. Each stop also shifts lightness, not just hue, for the same
// colour-blindness/projector reason as the risk bands above.
const THERMAL_STOPS = [
  { stop: 0.0, rgb: [30, 58, 138] }, // deep blue, dark
  { stop: 0.33, rgb: [8, 145, 178] }, // teal, mid
  { stop: 0.66, rgb: [217, 119, 6] }, // amber, bright
  { stop: 1.0, rgb: [220, 38, 38] }, // red, dark-bright
];

function rgbToHex([r, g, b]) {
  return `#${[r, g, b].map((c) => Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2, "0")).join("")}`;
}

/** value in [min, max] -> a hex colour along the thermal scale, clamped at the ends. */
export function thermalColor(value, min, max) {
  if (!Number.isFinite(value) || max <= min) return rgbToHex(THERMAL_STOPS[0].rgb);
  const t = Math.min(1, Math.max(0, (value - min) / (max - min)));

  for (let i = 0; i < THERMAL_STOPS.length - 1; i++) {
    const a = THERMAL_STOPS[i];
    const b = THERMAL_STOPS[i + 1];
    if (t >= a.stop && t <= b.stop) {
      const localT = (t - a.stop) / (b.stop - a.stop);
      const rgb = a.rgb.map((c, idx) => c + (b.rgb[idx] - c) * localT);
      return rgbToHex(rgb);
    }
  }
  return rgbToHex(THERMAL_STOPS[THERMAL_STOPS.length - 1].rgb);
}

export const THERMAL_SCALE_STOPS = THERMAL_STOPS.map((s) => ({ stop: s.stop, color: rgbToHex(s.rgb) }));
