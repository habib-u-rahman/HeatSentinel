"""Batch surface-composition analysis over a folder of street-level images.

Usage:
    python scripts/analyze_images.py
    python scripts/analyze_images.py --images-dir ../data/street_images --manifest manifest.csv
    python scripts/analyze_images.py --limit 3 --overlay-dir ../data/overlays

Coordinates come from a CSV manifest (filename,lat,lon[,point_id,...]) if
present, falling back to GPS EXIF tags per image. Images with neither are
skipped and reported. When the manifest carries a point_id column (as written
by scripts/fetch_images.py), it is carried through into the output parquet so
rows can be joined back to data/raw/sample_points.parquet.

Every image is cached by the sha256 of its bytes: a second run over the same
files does zero model inference. Progress is saved after each new image, so a
killed/interrupted run can simply be re-run to resume.

Different point_ids can legitimately share the SAME image content -- e.g.
fetch_images.py matching several nearby sample points to the same nearest
Mapillary photo in a sparsely-covered area. A sha256 cache hit for a point_id
we haven't recorded yet still gets its own output row (reusing the cached
inference result, at zero extra cost) -- otherwise those points would silently
vanish from the training set instead of just sharing a photo.

Run from the backend/ directory so the app package is importable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from app.vision import segmenter  # noqa: E402
from app.vision.pipeline import analyze_image  # noqa: E402

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(manifest_path: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries[row["filename"]] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "point_id": row.get("point_id") or None,
            }
    return entries


def _dms_to_degrees(dms, ref) -> Optional[float]:
    try:
        degrees = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1]
        seconds = dms[2][0] / dms[2][1]
    except (IndexError, ZeroDivisionError):
        return None
    value = degrees + minutes / 60.0 + seconds / 3600.0
    ref_str = ref.decode() if isinstance(ref, bytes) else ref
    if ref_str in ("S", "W"):
        value = -value
    return value


def _gps_from_exif(path: Path) -> Optional[tuple[float, float]]:
    try:
        import piexif
    except ImportError:
        return None

    try:
        exif_dict = piexif.load(str(path))
    except Exception:
        return None

    gps = exif_dict.get("GPS")
    if not gps:
        return None

    lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)
    if not (lat_dms and lat_ref and lon_dms and lon_ref):
        return None

    lat = _dms_to_degrees(lat_dms, lat_ref)
    lon = _dms_to_degrees(lon_dms, lon_ref)
    if lat is None or lon is None:
        return None
    return lat, lon


def _resolve_coords(path: Path, manifest: dict[str, dict]) -> Optional[tuple[float, float, Optional[str]]]:
    entry = manifest.get(path.name)
    if entry is not None:
        return entry["lat"], entry["lon"], entry["point_id"]
    exif_coords = _gps_from_exif(path)
    if exif_coords is None:
        return None
    lat, lon = exif_coords
    return lat, lon, None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Batch-analyze street images for surface composition")
    parser.add_argument("--images-dir", default="../data/street_images", help="folder of input images")
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV with filename,lat,lon (default: <images-dir>/manifest.csv if present; else GPS EXIF)",
    )
    parser.add_argument("--output", default="../data/raw/surface_profiles.parquet", help="output parquet path")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N images")
    parser.add_argument("--overlay-dir", default=None, help="folder to write colourised segmentation overlays")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else images_dir / "manifest.csv"
    manifest = _load_manifest(manifest_path) if manifest_path.exists() else {}

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    existing_df = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
    all_rows: list[dict] = existing_df.to_dict("records") if not existing_df.empty else []
    cached_row_by_sha256: dict[str, dict] = {}
    seen_point_ids: set = set()
    for record in all_rows:
        sha = record.get("sha256")
        if sha:
            cached_row_by_sha256[sha] = record
        if record.get("point_id") is not None:
            seen_point_ids.add(record["point_id"])

    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    skipped_no_coords: list[str] = []
    cache_hits = 0
    reused_count = 0
    new_count = 0
    overlays_generated = 0

    for path in tqdm(image_paths, desc="analyzing images"):
        sha256 = _sha256_of_file(path)
        coords = _resolve_coords(path, manifest)
        if coords is None:
            skipped_no_coords.append(path.name)
            continue
        lat, lon, point_id = coords

        # Overlay generation is independent of the inference cache: an image's
        # NUMERIC results (buckets/metrics) are cached, but the per-pixel mask
        # needed to draw an overlay never was, so a cache-hit image still needs
        # its own segmentation pass to produce an overlay -- gate on whether the
        # overlay FILE already exists, not on whether inference already ran.
        overlay_dest = (overlay_dir / f"{path.stem}_overlay.png") if overlay_dir else None
        needs_overlay = overlay_dest is not None and not overlay_dest.exists()

        if sha256 in cached_row_by_sha256:
            cache_hits += 1
            if needs_overlay:
                segmenter.save_overlay(path, overlay_dest)
                overlays_generated += 1
            if point_id is not None and point_id not in seen_point_ids:
                reused_row = dict(cached_row_by_sha256[sha256])
                reused_row.update(point_id=point_id, image_path=str(path), lat=lat, lon=lon)
                all_rows.append(reused_row)
                seen_point_ids.add(point_id)
                reused_count += 1
                pd.DataFrame(all_rows).to_parquet(output_path, index=False)
            continue

        profile = analyze_image(path, lat, lon)
        row = profile.model_dump()
        row["sha256"] = sha256
        row["point_id"] = point_id

        if needs_overlay:
            segmenter.save_overlay(path, overlay_dest)
            overlays_generated += 1

        all_rows.append(row)
        cached_row_by_sha256[sha256] = row
        if point_id is not None:
            seen_point_ids.add(point_id)
        new_count += 1
        # Save after every image so a killed run can simply be re-run to resume.
        pd.DataFrame(all_rows).to_parquet(output_path, index=False)

    print(
        f"processed: {new_count} new (real inference), {reused_count} reused (duplicate image, new point_id), "
        f"{cache_hits - reused_count} fully cached (same point_id already recorded), "
        f"{len(skipped_no_coords)} skipped (no coords)"
    )
    if overlay_dir:
        print(f"overlays generated: {overlays_generated} (skipped if already present in {overlay_dir})")
    if skipped_no_coords:
        print("skipped (no manifest entry or GPS EXIF):")
        for name in skipped_no_coords:
            print(f"  - {name}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
