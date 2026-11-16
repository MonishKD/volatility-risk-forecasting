"""Download raw price snapshots for the configured symbols."""

import argparse

from vol_forecast.config import load_config
from vol_forecast.data import download_snapshot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite existing snapshots")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for symbol in cfg.symbols:
        path = download_snapshot(
            symbol, cfg.data_start, cfg.data_end, cfg.auto_adjust, cfg.raw_dir, force=args.force
        )
        print(f"{symbol}: {path}")
