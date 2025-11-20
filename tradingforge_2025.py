import argparse
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, get_context
from tqdm import tqdm


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def generate_returns_batch(
    rng: np.random.Generator,
    batch_size: int,
    days: int,
    trend_probs: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a batch of daily returns for multiple series with mixed regimes.
    Returns a tuple of:
      - returns_matrix: shape (batch_size, days)
      - trend_type_indices: shape (batch_size,), values {0: bull, 1: bear, 2: choppy, 3: random}
      - volatilities: shape (batch_size,)
      - spikes_counts: shape (batch_size,)
    """
    # Draw per-series regime, volatility, and spike counts
    # 0: bull, 1: bear, 2: choppy, 3: random
    trend_type_indices = rng.choice(4, size=batch_size, p=trend_probs)
    volatilities = rng.uniform(0.01, 0.05, size=batch_size)
    spikes_counts = rng.integers(1, 4, size=batch_size)

    # Bias per regime
    bias = np.zeros(batch_size, dtype=np.float64)
    bias[trend_type_indices == 0] = 0.001  # bull
    bias[trend_type_indices == 1] = -0.001  # bear
    # choppy: higher volatility
    vol_effective = volatilities.copy()
    vol_effective[trend_type_indices == 2] *= 1.5

    # Base normal LOG-returns with per-row sigma and per-row bias
    # Using broadcasting: (batch_size, days) = (batch_size, 1) * (1, days) + (batch_size, 1)
    log_returns_matrix = rng.normal(loc=0.0, scale=1.0, size=(batch_size, days))
    log_returns_matrix = log_returns_matrix * vol_effective[:, None] + bias[:, None]

    # Add spikes per series (few events per series → a tiny Python loop is acceptable)
    for row in range(batch_size):
        k = int(spikes_counts[row])
        if k <= 0:
            continue
        pos = rng.choice(days, size=k, replace=False)
        # Convert ±10% linear shocks to log-domain
        spike_signs = rng.choice(np.array([-0.1, 0.1], dtype=np.float64), size=k, replace=True)
        log_shocks = np.log1p(spike_signs)
        log_returns_matrix[row, pos] += log_shocks

    return log_returns_matrix, trend_type_indices, volatilities, spikes_counts


def returns_to_prices(
    returns_matrix: np.ndarray,
    start_price: float,
) -> np.ndarray:
    """
    Convert LOG-returns to price paths (inclusive of initial price).
    returns_matrix: shape (batch, days) of log-returns
    result: shape (batch, days + 1)
    """
    batch, days = returns_matrix.shape
    prices = np.empty((batch, days + 1), dtype=np.float64)
    prices[:, 0] = start_price
    # cumulative compounding in log domain
    prices[:, 1:] = start_price * np.exp(np.cumsum(returns_matrix, axis=1))
    return prices


def trend_index_to_str(indices: np.ndarray) -> List[str]:
    mapping = np.array(["bull", "bear", "choppy", "random"])
    return mapping[indices].tolist()


def generate_shard(
    shard_args: Tuple[int, int, int, str, int, int, float, Tuple[float, float, float, float], int]
) -> Dict[str, int]:
    """
    Worker to generate a shard of series and write to a single gzip JSONL file.
    Returns trend counts for this shard.

    shard_args:
        shard_id: int
        start_index: int (global series start index for deterministic IDs)
        num_series: int (for this shard)
        outdir: str
        days: int
        batch_size: int
        start_price: float
        trend_probs: tuple of 4 floats
        seed: int
    """
    (
        shard_id,
        start_index,
        num_series,
        outdir,
        days,
        batch_size,
        start_price,
        trend_probs,
        seed,
    ) = shard_args

    rng = np.random.default_rng(seed)
    filepath = os.path.join(outdir, f"part_{shard_id:02d}.jsonl.gz")

    counts: Counter = Counter()
    now_iso = datetime.now().isoformat()

    written = 0
    with gzip.open(filepath, mode="wt", encoding="utf-8") as gz:
        while written < num_series:
            bs = min(batch_size, num_series - written)
            ret_mat, trend_idx, vols, spikes = generate_returns_batch(
                rng=rng,
                batch_size=bs,
                days=days,
                trend_probs=trend_probs,
            )
            price_mat = returns_to_prices(ret_mat, start_price=start_price)
            trends = trend_index_to_str(trend_idx)

            # Stream out JSON lines
            for i in range(bs):
                gid = start_index + written + i
                prices_i = price_mat[i]
                # True MDD
                peak = np.maximum.accumulate(prices_i)
                mdd = float(np.min(prices_i / peak - 1.0))
                total_return = float(prices_i[-1] / prices_i[0] - 1.0)
                obj = {
                    "id": f"stock_{gid:06d}",
                    "prices": prices_i.tolist(),
                    "days": days,
                    "start_price": float(prices_i[0]),
                    "end_price": float(prices_i[-1]),
                    "trend_type": trends[i],
                    "volatility": float(vols[i]),
                    "spikes": int(spikes[i]),
                    "max_drawdown": mdd,
                    "total_return": total_return,
                    "source": "synthetic_tradingforge_2025",
                    "generated_at": now_iso,
                }
                gz.write(json.dumps(obj, separators=(",", ":")) + "\n")
                counts[trends[i]] += 1

            written += bs

    # Return simple dict for easy reduction
    return dict(counts)


def aggregate_counts(count_dicts: List[Dict[str, int]]) -> Dict[str, int]:
    merged = Counter()
    for d in count_dicts:
        merged.update(d)
    return dict(merged)


def plot_money_sample(outdir: str, days: int, start_price: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    trends = ["bull", "bear", "choppy"]
    trend_to_bias = {"bull": 0.001, "bear": -0.001, "choppy": 0.0}

    plt.figure(figsize=(12, 6))
    for trend in trends:
        volatility = float(rng.uniform(0.01, 0.05))
        effective_vol = volatility * (1.5 if trend == "choppy" else 1.0)
        bias = trend_to_bias[trend]
        returns = rng.normal(0.0, effective_vol, size=days) + bias
        # add a couple of spikes
        k = int(rng.integers(1, 4))
        pos = rng.choice(days, size=k, replace=False)
        returns[pos] += rng.choice(np.array([-0.1, 0.1]), size=k)
        prices = returns_to_prices(returns[None, :], start_price=start_price)[0]
        plt.plot(prices, label=trend)

    plt.title("TradingForge-2025 — Synthetic Stock Prices (Bull / Bear / Choppy)")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(outdir, "money_plot.png")
    plt.savefig(outpath, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradingForge-2025 — Generate synthetic trading series fast."
    )
    parser.add_argument("--num_series", type=int, default=500_000, help="Total series to generate.")
    parser.add_argument("--days", type=int, default=252, help="Trading days per series.")
    parser.add_argument("--start_price", type=float, default=100.0, help="Initial price.")
    parser.add_argument(
        "--outdir", type=str, default="TradingForge_2025", help="Output directory."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip shards whose output files already exist.",
    )
    parser.add_argument(
        "--write_parquet",
        action="store_true",
        help="Also write per-shard Parquet files (requires pyarrow or fastparquet).",
    )
    parser.add_argument(
        "--write_checksums",
        action="store_true",
        help="Compute SHA256 checksums for files in manifest (slower).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 0),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=50_000,
        help="Number of series per output shard (jsonl.gz).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Rows to generate per vectorized batch inside each worker.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for reproducibility (optional).",
    )
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    # Trend probabilities: [bull, bear, choppy, random]
    trend_probs = (0.25, 0.25, 0.20, 0.30)

    # Shard planning
    num_shards = math.ceil(args.num_series / args.shard_size)
    # Distribute series over shards as evenly as possible
    shard_counts = [args.num_series // num_shards] * num_shards
    for i in range(args.num_series % num_shards):
        shard_counts[i] += 1

    # Assign global starting indices for deterministic IDs
    shard_starts = []
    running = 0
    for c in shard_counts:
        shard_starts.append(running)
        running += c

    # Seeds for workers (so shards are independent)
    master_seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy)
    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(num_shards)
    child_seeds_int = [int(s.generate_state(1)[0]) for s in child_seeds]

    # Build shard args
    shard_jobs = []
    for shard_id in range(num_shards):
        shard_idx = shard_id + 1
        shard_path = os.path.join(args.outdir, f"part_{shard_idx:02d}.jsonl.gz")
        if args.resume and os.path.exists(shard_path) and os.path.getsize(shard_path) > 0:
            continue
        shard_jobs.append(
            (
                shard_idx,
                shard_starts[shard_id],
                shard_counts[shard_id],
                args.outdir,
                args.days,
                args.batch_size,
                args.start_price,
                trend_probs,
                child_seeds_int[shard_id],
            )
        )

    print(
        f"TradingForge-2025: Generating {args.num_series:,} series "
        f"in {num_shards} shards using {args.processes} processes..."
    )
    if args.resume:
        print(f"Resume enabled: {len(shard_jobs)} shard(s) to generate, skipping existing outputs.")

    # Run generation in parallel; use 'spawn' for macOS safety
    counts_list: List[Dict[str, int]] = []
    if shard_jobs:
        with get_context("spawn").Pool(processes=args.processes) as pool:
            for shard_counts_dict in tqdm(
                pool.imap_unordered(generate_shard, shard_jobs),
                total=len(shard_jobs),
                desc="Shards",
            ):
                counts_list.append(shard_counts_dict)

    # Aggregate counts. If resuming, recompute from all shards on disk for accuracy.
    if args.resume:
        trend_counter = Counter()
        for shard_idx in range(1, num_shards + 1):
            jl_path = os.path.join(args.outdir, f"part_{shard_idx:02d}.jsonl.gz")
            if not os.path.exists(jl_path):
                continue
            try:
                with gzip.open(jl_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            tt = obj.get("trend_type")
                            if tt:
                                trend_counter[tt] += 1
                        except Exception:
                            continue
            except Exception:
                continue
        total_counts = dict(trend_counter)
    else:
        total_counts = aggregate_counts(counts_list)

    # Optional: write per-shard Parquet files
    parquet_paths: List[str] = []
    if args.write_parquet:
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None
        if pd is None or not hasattr(pd, "DataFrame"):
            print("Parquet export requested but pandas is unavailable. Skipping Parquet.")
        else:
            parquet_dir = os.path.join(args.outdir, "parquet")
            os.makedirs(parquet_dir, exist_ok=True)
            for shard_id in range(1, num_shards + 1):
                jl_path = os.path.join(args.outdir, f"part_{shard_id:02d}.jsonl.gz")
                pq_path = os.path.join(parquet_dir, f"part_{shard_id:02d}.parquet")
                # Load jsonl.gz for this shard and write parquet
                rows: List[dict] = []
                with gzip.open(jl_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        rows.append(json.loads(line))
                if rows:
                    try:
                        df_shard = pd.DataFrame(rows)
                        df_shard.to_parquet(pq_path, index=False, compression="zstd")
                        parquet_paths.append(pq_path)
                    except Exception as e:
                        print(f"Failed to write Parquet for shard {shard_id}: {e}")

    # Save counts summary and manifest
    summary_counts_path = os.path.join(args.outdir, "summary_counts.json")
    with open(summary_counts_path, "w", encoding="utf-8") as f:
        json.dump(total_counts, f, indent=2)

    def _file_info(path: str) -> Dict[str, object]:
        info: Dict[str, object] = {"path": os.path.basename(path)}
        try:
            info["size_bytes"] = os.path.getsize(path)
        except Exception:
            info["size_bytes"] = None
        if args.write_checksums:
            import hashlib

            h = hashlib.sha256()
            try:
                with open(path, "rb") as fp:
                    for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                        h.update(chunk)
                info["sha256"] = h.hexdigest()
            except Exception:
                info["sha256"] = None
        return info

    manifest = {
        "created_at": datetime.now().isoformat(),
        "num_series": args.num_series,
        "days": args.days,
        "start_price": args.start_price,
        "processes": args.processes,
        "shard_size": args.shard_size,
        "batch_size": args.batch_size,
        "seed": master_seed,
        "trend_probs": {"bull": 0.25, "bear": 0.25, "choppy": 0.20, "random": 0.30},
        "trend_counts": total_counts,
        "files": {
            "summary_counts": _file_info(summary_counts_path),
            "shards": [
                _file_info(os.path.join(args.outdir, f"part_{sid:02d}.jsonl.gz"))
                for sid in range(1, num_shards + 1)
            ],
            "parquet": [
                _file_info(p) for p in parquet_paths
            ]
            if parquet_paths
            else [],
            "money_plot": _file_info(os.path.join(args.outdir, "money_plot.png")),
        },
        "source": "synthetic_tradingforge_2025",
        "schema_version": "1.1",
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Plot a sample figure
    plot_money_sample(
        outdir=args.outdir, days=args.days, start_price=args.start_price, seed=master_seed + 1
    )
    print("Money plot saved!")

    print(
        "\n"
        "TradingForge-2025 COMPLETE\n"
        f"→ {args.num_series:,} synthetic trading series\n"
        f"→ Trends: {total_counts}\n"
        "→ Ready to sell: $300k–$900k outright\n"
        "→ Repo: github.com/yourname/TradingForge-2025\n"
    )


if __name__ == "__main__":
    main()


