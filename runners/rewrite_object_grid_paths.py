#!/usr/bin/env python3

import argparse
from pathlib import Path

try:
    from cedarkit.utils.io.cloudjoblib import joblib_cloud_load, joblib_cloud_atomic_dump
except ImportError as import_error:
    try:
        from utils.io.cloudjoblib import joblib_cloud_load, joblib_cloud_atomic_dump
    except ImportError:
        raise import_error


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite object-grid embedded output paths to a new dyad-home/tmp-home template."
    )
    parser.add_argument(
        "--input-grid",
        required=True,
        type=Path,
        help="Path to existing object grid joblib.",
    )
    parser.add_argument(
        "--output-grid",
        type=Path,
        default=None,
        help="Path for rewritten object grid joblib. Required unless --in-place is used.",
    )
    parser.add_argument(
        "--new-dyad-home",
        required=True,
        type=Path,
        help="Path to parent directory that contains dyad folders.",
    )
    parser.add_argument(
        "--tmp-home",
        required=True,
        type=str,
        help="Dyad folder name used to build <new-dyad-home>/<tmp-home>/tmp.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input object grid file in place.",
    )
    return parser.parse_args()


def _resolve_output_path(args):
    if args.in_place and args.output_grid is not None:
        raise ValueError("Use either --in-place or --output-grid, not both.")
    if args.in_place:
        return args.input_grid
    if args.output_grid is None:
        raise ValueError("Provide --output-grid when not using --in-place.")
    return args.output_grid


def rewrite_object_grid_paths(input_grid, output_grid, new_dyad_home, tmp_home):
    object_grid = joblib_cloud_load(str(input_grid))

    migrated = 0
    skipped = 0
    for key, cell_obj in object_grid.items():
        if cell_obj is None:
            skipped += 1
            continue
        output_obj = getattr(cell_obj, "output", None)
        if output_obj is None or not hasattr(output_obj, "migrate_path"):
            skipped += 1
            continue
        output_obj.migrate_path(new_dyad_home=new_dyad_home, tmp_home=tmp_home)
        migrated += 1

    output_grid.parent.mkdir(parents=True, exist_ok=True)
    joblib_cloud_atomic_dump(object_grid, str(output_grid))
    return migrated, skipped, len(object_grid)


def main():
    args = _parse_args()
    output_grid = _resolve_output_path(args)

    if not args.input_grid.exists():
        raise FileNotFoundError(f"Input object grid does not exist: {args.input_grid}")

    migrated, skipped, total = rewrite_object_grid_paths(
        input_grid=args.input_grid,
        output_grid=output_grid,
        new_dyad_home=Path(args.new_dyad_home),
        tmp_home=args.tmp_home,
    )

    print(f"Rewrote object grid: {output_grid}")
    print(f"Entries total={total}, migrated={migrated}, skipped={skipped}")
    print(f"Template root: {Path(args.new_dyad_home) / args.tmp_home / 'tmp'}")


if __name__ == "__main__":
    main()
