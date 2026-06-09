"""
feasibility.py - deployment-feasibility profiling of the IE+IR+ER pipeline.

The pipeline offloads all heavy compute (embeddings, LLM) to a remote API;
locally it only runs FAISS HNSW search on CPU. This script quantifies what a
production deployment actually needs, across three axes:

  (1) Streaming throughput + cost amortization (per dataset)
      Stream N unique descriptions through the full IE+IR+ER pipeline with a
      catalog seeded from DVM-CAR, in prefetched batches. Per batch we record
      per-stage latency, LLM/embedding call counts, tokens, cache-hit rate,
      catalog growth, and process RSS. The headline result is that LLM calls
      per record amortizes toward the 1-call extraction floor as the catalog
      saturates and the match cache fills.

  (2) Online (serial) latency
      A held-out slice processed one record at a time against a cold, growing
      catalog, giving the per-record end-to-end latency distribution (p50/p95)
      broken down by stage - the upper bound for interactive use.

  (3) Local compute footprint (no network)
      FAISS HNSW search latency / QPS on CPU over the seeded indexes, index
      memory + disk footprint, CPU utilization during streaming, peak RSS, and
      explicit confirmation that no GPU is used.

Writes:
  data/feasibility.json
  manuscript/figures/feasibility.pdf, .png

Usage:
    uv run python scripts/feasibility.py
    uv run python scripts/feasibility.py --n 5000 --datasets autoscout24 mucars
    uv run python scripts/feasibility.py --plot_only
"""

import os

# faiss and matplotlib/numpy can each link a copy of libomp; allow the duplicate
# rather than abort. Must be set before faiss is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import asyncio
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
from src.normalizer import Normalizer, serialize_row, _EMBED_DIM  # noqa: E402
from scripts.evaluate_normalizer import _fresh_catalog, clip_row  # noqa: E402
from scripts.ablation_threshold import COLORS, _setup_rc  # noqa: E402

load_dotenv()

DEFAULT_DATASETS = ["autoscout24", "mucars"]
DEFAULT_N = 5000
BATCH_SIZE = 25
PREFETCH = 1
FAISS_QUERIES = 5000    # random queries per index for the search micro-benchmark
MAX_CONCURRENCY = 50
REQUEST_TIMEOUT = 30.0  # per-request API timeout; stalls become retryable
BATCH_TIMEOUT = 240.0   # watchdog: skip a batch that exceeds this many seconds

DATASETS = {
    "autoscout24": "data/autoscout24.csv",
    "mucars": "data/mucars.csv",
}


# helpers

def _pct(xs: List[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def _summary(xs: List[float]) -> dict:
    return {
        "mean": float(np.mean(xs)) if xs else float("nan"),
        "p50": _pct(xs, 50),
        "p95": _pct(xs, 95),
        "max": float(np.max(xs)) if xs else float("nan"),
    }


def load_descriptions(path: str, n: int) -> List[str]:
    """Unique, serialized descriptions over all columns (matches the eval setup)."""
    df = pd.read_csv(path, low_memory=False)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    df = df.drop_duplicates()
    descs: List[str] = []
    seen: set = set()
    for _, row in df.iterrows():
        s = serialize_row(clip_row(row))
        if s and s not in seen:
            seen.add(s)
            descs.append(s)
        if len(descs) >= n:
            break
    return descs


def _cpu_brand() -> str:
    """Human-readable CPU/chip string (e.g. 'Apple M4'), best-effort."""
    try:
        import subprocess
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def capture_env() -> dict:
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": _cpu_brand(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_gb": round(vm.total / 1e9, 1),
        "faiss_num_gpus": faiss.get_num_gpus(),
        "gpu_used": faiss.get_num_gpus() > 0,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
    }


def _wrap_embedding_counter(norm: Normalizer) -> dict:
    """Wrap the async embeddings.create to count embedding API calls + inputs."""
    counter = {"calls": 0, "inputs": 0}
    orig = norm.openai_client.embeddings.create

    async def wrapped(*args, **kwargs):
        counter["calls"] += 1
        inp = kwargs.get("input")
        counter["inputs"] += len(inp) if isinstance(inp, list) else 1
        return await orig(*args, **kwargs)

    norm.openai_client.embeddings.create = wrapped
    return counter


# (3) FAISS HNSW search micro-benchmark (local, no network)

def faiss_microbench(seed_dir: str, embedding_model: str, k: int,
                     ef_search: int, n_queries: int) -> dict:
    dim = _EMBED_DIM[embedding_model]
    proc = psutil.Process()
    out: dict = {"attributes": {}}
    per_record_p50 = 0.0
    per_record_p95 = 0.0

    for attr in ["brand", "model"]:
        index_path = os.path.join(seed_dir, f"hnsw_{attr}.index")
        if not os.path.exists(index_path):
            continue
        rss_before = proc.memory_info().rss
        idx = faiss.read_index(index_path)
        idx.hnsw.efSearch = ef_search
        rss_after = proc.memory_info().rss

        # Random unit query vectors (no network) - measures CPU traversal cost.
        rng = np.random.default_rng(0)
        q = rng.standard_normal((n_queries, dim)).astype(np.float32)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        kk = min(k, idx.ntotal)

        # Warm-up, then time per-query (single-vector search = production path).
        idx.search(q[:50], kk)
        lat_ms: List[float] = []
        t_all = time.perf_counter()
        for i in range(n_queries):
            t0 = time.perf_counter()
            idx.search(q[i:i + 1], kk)
            lat_ms.append((time.perf_counter() - t0) * 1e3)
        wall = time.perf_counter() - t_all

        s = _summary(lat_ms)
        per_record_p50 += s["p50"]
        per_record_p95 += s["p95"]
        out["attributes"][attr] = {
            "ntotal": int(idx.ntotal),
            "dim": dim,
            "disk_bytes": os.path.getsize(index_path),
            "ram_bytes_loaded": int(rss_after - rss_before),
            "search_ms": s,
            "qps": n_queries / wall,
        }

    # A record retrieves both brand and model, so per-record search cost sums.
    out["per_record_search_ms_p50"] = per_record_p50
    out["per_record_search_ms_p95"] = per_record_p95
    return out


# (1) streaming throughput + amortization

async def stream_dataset(listings: List[str], seed_dir: str, dataset: str,
                         batch_size: int, prefetch: int) -> dict:
    tmp_dir = _fresh_catalog(seed_dir, f"feas_{dataset}")
    proc = psutil.Process()
    try:
        norm = Normalizer(match_mode="llm", persist_directory=tmp_dir,
                           max_concurrency=MAX_CONCURRENCY,
                           request_timeout=REQUEST_TIMEOUT)
        emb = _wrap_embedding_counter(norm)
        skipped = 0

        batches = [listings[i:i + batch_size]
                   for i in range(0, len(listings), batch_size)]

        trace: List[dict] = []
        cum_records = 0
        peak_rss = proc.memory_info().rss
        cpu0 = proc.cpu_times()
        t_start = time.perf_counter()

        # Time the extraction stage so it is not lost from per-record latency:
        # under prefetch, extract_all runs as a background task and its wall
        # time would otherwise be excluded from the pipeline's own timer.
        async def _timed_extract(b):
            s = time.perf_counter()
            res = await norm.extract_all(b)
            return res, time.perf_counter() - s

        prefetch_tasks: Dict[int, asyncio.Task] = {}
        for j in range(min(prefetch, len(batches))):
            prefetch_tasks[j] = asyncio.create_task(_timed_extract(batches[j]))

        lat_samples: List[float] = []   # per-record latency (record-weighted)
        prev_done = t_start
        for idx, batch in enumerate(batches):
            ahead = idx + prefetch
            if ahead < len(batches):
                prefetch_tasks[ahead] = asyncio.create_task(
                    _timed_extract(batches[ahead]))
            if idx in prefetch_tasks:
                pre, extract_s = await prefetch_tasks.pop(idx)
            else:
                pre, extract_s = None, 0.0

            try:
                _, m = await asyncio.wait_for(
                    norm(batch, extractions=pre), timeout=BATCH_TIMEOUT)
            except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
                skipped += 1
                print(f"    [warn] batch {idx} skipped ({type(e).__name__}); "
                      f"{skipped} skipped so far", flush=True)
                continue
            t_done = time.perf_counter()
            cum_records += len(batch)
            rss = proc.memory_info().rss
            peak_rss = max(peak_rss, rss)

            # End-to-end per-record latency: extraction + retrieval + resolution
            # + dedup. All records in a batch complete together, so each is
            # assigned the batch's latency (record-weighted for percentiles).
            latency_s = extract_s + m["total_s"]
            lat_samples.extend([latency_s] * len(batch))
            # Realized throughput: records over the wall time between completions.
            batch_wall = max(t_done - prev_done, 1e-9)
            dpm_inst = len(batch) / batch_wall * 60.0
            prev_done = t_done

            cum_extract = norm.tokens["extract"]["count"]
            cum_match = norm.tokens["match"]["count"]
            trace.append({
                "batch": idx,
                "cum_records": cum_records,
                "latency_s": latency_s,
                "extract_s": extract_s,
                "retrieve_s": m["retrieve_s"],
                "match_s": m["match_s"],
                "dedup_s": m["dedup_s"],
                "proc_s": m["total_s"],
                "dpm_inst": dpm_inst,
                "match_calls": m["match_calls"],
                "cache_hits": m["cache_hits"],
                "cum_extract_calls": cum_extract,
                "cum_match_calls": cum_match,
                "cum_llm_calls": cum_extract + cum_match,
                "cum_llm_per_record": (cum_extract + cum_match) / cum_records,
                "catalog_brand": m["catalog_brand"],
                "catalog_model": m["catalog_model"],
                "rss_mb": rss / 1e6,
            })

        wall = time.perf_counter() - t_start
        cpu1 = proc.cpu_times()

        t = norm.tokens
        llm_calls = t["extract"]["count"] + t["match"]["count"]
        total_tokens = (t["extract"]["prompt"] + t["extract"]["completion"]
                        + t["match"]["prompt"] + t["match"]["completion"])
        n = cum_records
        cpu_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)

        # Steady-state marginal: LLM calls/record over the last 25% of the run.
        tail = trace[max(0, int(len(trace) * 0.75)):]
        tail_records = tail[-1]["cum_records"] - tail[0]["cum_records"] + len(batches[0])
        tail_match = tail[-1]["cum_match_calls"] - tail[0]["cum_match_calls"]
        steady_llm_per_record = 1.0 + tail_match / max(tail_records, 1)

        summary = {
            "n": n,
            "wall_s": wall,
            "dpm": n / (wall / 60),
            "rpm": llm_calls / (wall / 60),
            "tpm": total_tokens / (wall / 60),
            "total_llm_calls": llm_calls,
            "extract_calls": t["extract"]["count"],
            "match_calls": t["match"]["count"],
            "llm_calls_per_record": llm_calls / n,
            "steady_llm_calls_per_record": steady_llm_per_record,
            "embedding_calls": emb["calls"],
            "embedding_inputs": emb["inputs"],
            "total_tokens": total_tokens,
            "tokens_per_record": total_tokens / n,
            "cost_usd": norm.get_cost(),
            "cost_per_record": norm.get_cost() / n,
            "latency_p50_s": _pct(lat_samples, 50),
            "latency_p95_s": _pct(lat_samples, 95),
            "latency_mean_s": float(np.mean(lat_samples)) if lat_samples else 0.0,
            "final_cache_hit_rate": trace[-1]["cache_hits"] / batch_size if trace else 0.0,
            "peak_rss_mb": peak_rss / 1e6,
            "cpu_seconds": cpu_s,
            "cpu_utilization": cpu_s / wall if wall else 0.0,
            "catalog_brand": trace[-1]["catalog_brand"] if trace else 0,
            "catalog_model": trace[-1]["catalog_model"] if trace else 0,
            "batches_skipped": skipped,
        }
        return {"trace": trace, "summary": summary}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# plotting

def plot_report(report: dict, out_pdf: str, out_png: str) -> None:
    _setup_rc()
    datasets = list(report["datasets"].keys())
    n_rows = len(datasets)
    pretty = {"autoscout24": "AutoScout24", "mucars": "MuCars"}

    fig, axes = plt.subplots(n_rows, 3, figsize=(9.2, 1.15 * n_rows + 0.8))
    if n_rows == 1:
        axes = np.array([axes])

    def _smooth(v, w=9):
        v = np.asarray(v, dtype=float)
        if len(v) < 3:
            return v
        w = min(w, len(v) if len(v) % 2 else len(v) - 1)
        if w < 3:
            return v
        pad = w // 2
        vp = np.pad(v, pad, mode="edge")
        return np.convolve(vp, np.ones(w) / w, mode="valid")

    for row, ds in enumerate(datasets):
        trace = report["datasets"][ds]["stream"]["trace"]
        x = [t["cum_records"] for t in trace]

        # Col 0: end-to-end per-record latency (extraction + resolution)
        ax = axes[row, 0]
        lat = [t["latency_s"] for t in trace]
        ax.plot(x, _smooth(lat), lw=1.4, color=COLORS["blue_dark"], zorder=3)
        if row == 0:
            ax.set_title("Latency", pad=6)
        ax.set_ylabel("Latency (s)", fontsize=9.5, labelpad=3)
        ax.grid(True)
        ax.set_ylim(bottom=0)

        # Col 1: realized throughput over a trailing window
        # Per-batch instantaneous rate is bursty under prefetch (back-to-back
        # completions); average over a trailing window of the reconstructed
        # wall-clock timeline for a faithful records/min curve.
        ax = axes[row, 1]
        xr = np.array([0] + x, dtype=float)            # cumulative records
        dr = np.diff(xr)
        dpm_inst = np.array([t["dpm_inst"] for t in trace], dtype=float)
        batch_wall = dr / np.maximum(dpm_inst, 1e-9) * 60.0
        cw = np.concatenate([[0.0], np.cumsum(batch_wall)])  # cumulative seconds
        K = 12
        thr = []
        for i in range(1, len(xr)):
            j = max(0, i - K)
            dt = cw[i] - cw[j]
            thr.append((xr[i] - xr[j]) / dt * 60.0 if dt > 1e-9 else 0.0)
        ax.plot(x, thr, lw=1.4, color=COLORS["green_dark"], zorder=3)
        if row == 0:
            ax.set_title("Throughput", pad=6)
        ax.set_ylabel("Records / min", fontsize=9.5, labelpad=3)
        ax.grid(True)
        ax.set_ylim(bottom=0)

        # Col 2: API calls per record, decaying to extraction floor
        ax = axes[row, 2]
        cum = [t["cum_llm_per_record"] for t in trace]
        ax.plot(x, cum, lw=1.4, color=COLORS["gold_dark"], zorder=3)
        ax.axhline(1.0, color=COLORS["ie_ref"], ls=(0, (4, 3)), lw=0.9,
                   alpha=0.75, zorder=1)
        if row == 0:
            ax.set_title("API calls", pad=6)
        ax.set_ylabel("Calls / record", fontsize=9.5, labelpad=3)
        ax.grid(True)
        ax.set_ylim(bottom=0.9)

        for c in range(3):
            if row == n_rows - 1:
                axes[row, c].set_xlabel("Records processed")

    fig.tight_layout(h_pad=0.6, w_pad=1.5, rect=(0.035, 0.02, 1, 1))

    for row, ds in enumerate(datasets):
        bbox = axes[row, 0].get_position()
        fig.text(0.008, (bbox.y0 + bbox.y1) / 2.0, pretty.get(ds, ds),
                 fontsize=11, fontweight="bold", color="#222222",
                 ha="left", va="center", rotation=90)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


# entrypoint

def _write_report(args, env, faiss_bench, per_ds, k, ef_search, embedding_model):
    report = {
        "env": env,
        "config": {"n": args.n, "batch_size": args.batch_size,
                   "prefetch": args.prefetch,
                   "max_concurrency": MAX_CONCURRENCY, "k": k,
                   "ef_search": ef_search, "embedding_model": embedding_model,
                   "model": "gpt-4.1-nano", "match_model": "gpt-4.1-mini",
                   "request_timeout": REQUEST_TIMEOUT, "seed_dir": args.seed_dir},
        "faiss": faiss_bench,
        "datasets": per_ds,
    }
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: None)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deployment-feasibility profiling for IE+IR+ER.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=DEFAULT_DATASETS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--prefetch", type=int, default=PREFETCH)
    parser.add_argument("--faiss_queries", type=int, default=FAISS_QUERIES)
    parser.add_argument("--seed_dir", default="experiments/db")
    parser.add_argument("--out_json", default="data/feasibility.json")
    parser.add_argument("--out_pdf", default="manuscript/figures/feasibility.pdf")
    parser.add_argument("--out_png", default="manuscript/figures/feasibility.png")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)

    if args.plot_only:
        with open(args.out_json) as f:
            report = json.load(f)
        plot_report(report, args.out_pdf, args.out_png)
        return

    if not os.path.exists(os.path.join(args.seed_dir, "catalog.db")):
        print(f"[error] seed catalog not found at {args.seed_dir}/catalog.db")
        sys.exit(1)

    env = capture_env()
    print("Environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")

    # Config for catalog params (defaults match Normalizer).
    k, ef_search, embedding_model = 5, 64, "text-embedding-3-small"

    print(f"\n[FAISS micro-benchmark] {args.faiss_queries} queries/index ...")
    faiss_bench = faiss_microbench(args.seed_dir, embedding_model, k,
                                   ef_search, args.faiss_queries)
    for attr, a in faiss_bench["attributes"].items():
        print(f"  {attr}: ntotal={a['ntotal']}  "
              f"p50={a['search_ms']['p50']:.3f}ms  p95={a['search_ms']['p95']:.3f}ms  "
              f"{a['qps']:.0f} qps  ram={a['ram_bytes_loaded']/1e6:.1f}MB")

    async def run_all() -> dict:
        per_ds: dict = {}
        for ds in args.datasets:
            print(f"\n=== {ds} ===")
            listings = load_descriptions(DATASETS[ds], args.n)
            print(f"  loaded {len(listings)} unique descriptions")

            print(f"  [stream] {len(listings)} records, "
                  f"batch={args.batch_size}, prefetch={args.prefetch} ...")
            stream = await stream_dataset(listings, args.seed_dir, ds,
                                          args.batch_size, args.prefetch)
            s = stream["summary"]
            print(f"    {s['wall_s']:.0f}s  {s['dpm']:.0f} DPM  {s['rpm']:.0f} RPM  "
                  f"{s['tpm']/1e6:.2f}M TPM  ${s['cost_usd']:.3f}")
            print(f"    latency p50={s['latency_p50_s']:.2f}s p95={s['latency_p95_s']:.2f}s  "
                  f"LLM/record {s['llm_calls_per_record']:.3f} "
                  f"(steady {s['steady_llm_calls_per_record']:.3f})  "
                  f"peak RSS {s['peak_rss_mb']:.0f}MB")

            per_ds[ds] = {"n": len(listings), "stream": stream}
            # Checkpoint after each dataset so a later failure preserves results.
            _write_report(args, env, faiss_bench, per_ds, k, ef_search,
                          embedding_model)
            print(f"  [checkpoint] wrote {args.out_json} ({len(per_ds)} dataset(s))")
        return per_ds

    per_ds = asyncio.run(run_all())
    report = _write_report(args, env, faiss_bench, per_ds, k, ef_search,
                           embedding_model)
    print(f"\nwrote {args.out_json}")

    plot_report(report, args.out_pdf, args.out_png)


if __name__ == "__main__":
    main()
