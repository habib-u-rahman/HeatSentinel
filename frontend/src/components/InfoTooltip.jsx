import { useState } from "react";

/** A small "(i)" hover/focus affordance for a short plain-language
 * explanation -- e.g. "this number comes from a computer-vision model, not
 * a measurement." Reuses the same `.heatsentinel-tooltip` skin Leaflet
 * layers already use (see index.css), just applied to a plain popover here
 * instead of a Leaflet tooltip. Exists because the only prior affordance was
 * a native `title=` attribute, which is barely discoverable and unusable by
 * keyboard/touch -- this is hover- AND focus-triggered instead. */
export default function InfoTooltip({ text }) {
  const [open, setOpen] = useState(false);

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label="More info"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-slate-400 text-[9px] font-bold leading-none text-slate-500 hover:border-teal-500 hover:text-teal-600 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
      >
        i
      </button>
      {open && (
        // Opens DOWNWARD (top-full, not bottom-full): every current usage
        // sits in a header near the top of a scrollable panel (Sidebar's
        // content div), where an upward-opening popover would clip against
        // the scroll container's edge.
        <span
          role="tooltip"
          className="heatsentinel-tooltip absolute left-1/2 top-full z-10 mt-1.5 w-56 -translate-x-1/2 text-[11px] font-normal normal-case leading-relaxed tracking-normal"
        >
          {text}
        </span>
      )}
    </span>
  );
}
