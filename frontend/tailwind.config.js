/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Light theme: names kept identical (950 = page background, 600 =
        // borders/dividers) so no component's className needs to change --
        // only these hex values flip. 950 was near-black, is now near-white.
        base: {
          950: "#090d16",
          900: "#111827",
          850: "#1f2937",
          800: "#374151",
          700: "#4b5563",
          600: "#374151",
        },
        risk: {
          safe: "#16a34a",
          caution: "#ca8a04",
          danger: "#ea580c",
          critical: "#dc2626",
        },
        thermal: {
          cold: "#1e3a8a",
          cool: "#0891b2",
          warm: "#d97706",
          hot: "#dc2626",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      transitionDuration: {
        DEFAULT: "150ms",
      },
    },
  },
  plugins: [],
};
