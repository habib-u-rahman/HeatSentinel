"""Fetch one street-level image per sample point from Mapillary.

Reads data/raw/sample_points.parquet, downloads into data/street_images/, and
writes data/street_images/manifest.csv with columns
filename,lat,lon,point_id,image_id,captured_at -- exactly the format
scripts/analyze_images.py already expects (it ignores the extra columns).

Resumable: any point_id already present in manifest.csv is skipped without
hitting the network. Points with no imagery found within radius_m are
reported and written to data/raw/points_without_imagery.csv.

Run from the backend/ directory so the app package is importable.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from app.ingest.mapillary import MapillaryClient  # noqa: E402

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["filename", "lat", "lon", "point_id", "image_id", "captured_at"]


def _load_manifest_rows(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch street-level imagery for sample points")
    parser.add_argument("--sample-points", default="../data/raw/sample_points.parquet")
    parser.add_argument("--images-dir", default="../data/street_images")
    parser.add_argument("--manifest", default=None, help="default: <images-dir>/manifest.csv")
    parser.add_argument("--no-imagery-out", default="../data/raw/points_without_imagery.csv")
    parser.add_argument("--radius-m", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else images_dir / "manifest.csv"
    no_imagery_path = Path(args.no_imagery_out)
    no_imagery_path.parent.mkdir(parents=True, exist_ok=True)

    points_df = pd.read_parquet(args.sample_points)
    if args.limit is not None:
        points_df = points_df.head(args.limit)

    manifest_rows = _load_manifest_rows(manifest_path)
    done_point_ids = {row["point_id"] for row in manifest_rows}

    no_imagery: list[tuple[str, float, float]] = []
    fetched = 0
    skipped = 0

    with MapillaryClient() as client:
        for row in tqdm(points_df.itertuples(index=False), total=len(points_df), desc="fetching images"):
            point_id = row.point_id
            if point_id in done_point_ids:
                skipped += 1
                continue

            match = client.find_image_near(row.lat, row.lon, radius_m=args.radius_m)
            if match is None:
                no_imagery.append((point_id, row.lat, row.lon))
                continue

            filename = f"{point_id}_{match.image_id}.jpg"
            client.download_image(match.image_id, images_dir / filename)

            manifest_rows.append(
                {
                    "filename": filename,
                    "lat": match.lat,
                    "lon": match.lon,
                    "point_id": point_id,
                    "image_id": match.image_id,
                    "captured_at": match.captured_at.isoformat(),
                }
            )
            done_point_ids.add(point_id)
            fetched += 1
            # Save after every point so a killed run can simply be re-run to resume.
            _write_manifest(manifest_path, manifest_rows)

    if no_imagery:
        with no_imagery_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["point_id", "lat", "lon"])
            writer.writerows(no_imagery)

    print(f"fetched: {fetched} new, {skipped} already done, {len(no_imagery)} with no imagery found")
    print(f"manifest: {manifest_path}")
    if no_imagery:
        print(f"points without imagery written to: {no_imagery_path}")


if __name__ == "__main__":
    main()
