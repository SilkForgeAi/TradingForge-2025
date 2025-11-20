TradingForge-2025

**Institutional-grade synthetic trading data**  
1,000,000 price series · 252 trading days each

---Features
- Four market regimes: **bull / bear / choppy / random** (25 % | 25 % | 20 % | 30 %)
- Proper **log-returns** with realistic volatility and jump events
- Per-series metrics: `max_drawdown`, `total_return`, volatility, spike count
- **20 × GZIP-compressed JSONL shards** (50k series each)
- `manifest.json` + SHA256 checksums + `summary_counts.json`
- Parquet export available on request
- Fully reproducible (master seed + per-shard seeds)
- Generated on a laptop in under 15 minutes

---Output
