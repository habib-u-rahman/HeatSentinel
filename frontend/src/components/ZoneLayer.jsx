import { GeoJSON } from "react-leaflet";
import { riskColor } from "../utils/color";

/**
 * /api/zones as coloured semi-transparent rectangles. react-leaflet's
 * <GeoJSON> flips [lon,lat] -> [lat,lon] internally, so raw feature
 * coordinates are safe to pass straight through here (unlike RouteLayer,
 * which uses the low-level Polyline API and must flip manually).
 */
export default function ZoneLayer({ data, visible = true }) {
  if (!visible || !data?.features?.length) return null;

  const style = (feature) => {
    const color = riskColor(feature.properties.risk_band);
    return { color, weight: 1, fillColor: color, fillOpacity: 0.25 };
  };

  const onEachFeature = (feature, layer) => {
    const { zone_id, mean_wbgt_c, max_wbgt_c, risk_band, n_cells } = feature.properties;
    layer.bindTooltip(
      `<div class="label-caps">${zone_id} &middot; ${risk_band}</div>` +
        `<div class="num" style="font-size:0.9rem">${mean_wbgt_c.toFixed(1)}&deg;C mean / ${max_wbgt_c.toFixed(1)}&deg;C max</div>` +
        `<div style="color:#475569;font-size:0.7rem">${n_cells} grid cells</div>`,
      { sticky: true, className: "heatsentinel-tooltip" }
    );
    layer.on({
      mouseover: () => layer.setStyle({ fillOpacity: 0.45 }),
      mouseout: () => layer.setStyle({ fillOpacity: 0.25 }),
    });
  };

  // key forces a clean remount when the timestamp changes -- Leaflet's GeoJSON
  // layer doesn't diff new `data` against old on its own.
  return <GeoJSON key={data.observed_at} data={data} style={style} onEachFeature={onEachFeature} />;
}
