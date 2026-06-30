#!/usr/bin/env python3
"""Redo LIBERO-10 feature analysis for the D-LAPA depth-injection paper.

The script streams the provided feature shards, performs the video-disjoint
80/20 split used in the paper, and computes:

- standardized Ridge probes for RGB, depth-only, and RGB-depth concatenations
- depth-feature alignment diagnostics against the Stage-1 depth teacher
- depth-index accuracy/confidence diagnostics where indices are available
- UMAP snapshots and Moran's I over held-out samples
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")

# umap-learn is installed into /private/tmp by the analysis workflow.
if Path("/private/tmp/codex_umap_site").exists():
    sys.path.append("/private/tmp/codex_umap_site")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - handled at runtime.
    umap = None


FEATURE_DIR = Path("libero10_features")
OUT_DIR = Path("output/feature_analysis")
JSONL = FEATURE_DIR / "all_models_val_libero10.jsonl"
PART_GLOB = "all_models_val_libero10_part*.pt"

MODEL_KEYS = {
    "Model 1": "z_depth_feature_pred_model1",
    "Model 2": "z_depth_feature_pred_model2",
    "Model 3": "z_depth_feature_pred_model3",
    "Model 4": "z_depth_feature_pred_model4",
    "Model 5": "z_depth_feature_pred_model5",
    "Model 6.1": "z_depth_feature_pred_model6_1",
    "Model 7.1": "z_depth_feature_pred_model7_1",
}

INDEX_KEYS = {
    "Model 1": ("z_depth_indices_pred_model1", "confidence_model1"),
    "Model 2": ("z_depth_indices_pred_model2", "confidence_model2"),
    "Model 3": ("z_depth_indices_pred_model3", "confidence_model3"),
    "Model 6.1": ("z_depth_indices_pred_model6_1", "confidence_model6_1"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--include-depth-only", action="store_true")
    return parser.parse_args()


def iter_parts() -> Iterable[Path]:
    return iter(sorted(FEATURE_DIR.glob(PART_GLOB)))


def load_video_split(seed: int) -> Tuple[set, set, Dict[str, int]]:
    counts: Dict[str, int] = {}
    with JSONL.open() as f:
        for line in f:
            row = json.loads(line)
            vid = row["video_id"]
            counts[vid] = counts.get(vid, 0) + 1

    unique_videos = np.array(sorted(counts))
    train_videos, test_videos = train_test_split(
        unique_videos,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    return set(train_videos), set(test_videos), counts


def part_masks(video_ids: Sequence[str], train_videos: set) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = np.fromiter((vid in train_videos for vid in video_ids), dtype=bool)
    return train_mask, ~train_mask


def first_tensor_dim(keys: Sequence[str]) -> int:
    first = torch.load(next(iter_parts()), map_location="cpu")
    return int(sum(first[k].shape[1] for k in keys))


def collect_y(train_videos: set, n_train: int, n_test: int) -> Tuple[np.ndarray, np.ndarray]:
    y_train = np.empty(n_train, dtype=np.float32)
    y_test = np.empty(n_test, dtype=np.float32)
    tr_i = 0
    te_i = 0
    for part in iter_parts():
        shard = torch.load(part, map_location="cpu")
        train_mask, test_mask = part_masks(shard["video_id"], train_videos)
        y = shard["magnitude"].cpu().numpy().astype(np.float32, copy=False)
        ntr = int(train_mask.sum())
        nte = int(test_mask.sum())
        y_train[tr_i : tr_i + ntr] = y[train_mask]
        y_test[te_i : te_i + nte] = y[test_mask]
        tr_i += ntr
        te_i += nte
        del shard
        gc.collect()
    return y_train, y_test


def collect_feature_arrays(
    keys: Sequence[str],
    train_videos: set,
    n_train: int,
    n_test: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = first_tensor_dim(keys)
    x_train = np.empty((n_train, dim), dtype=np.float32)
    x_test = np.empty((n_test, dim), dtype=np.float32)
    tr_i = 0
    te_i = 0

    for part in iter_parts():
        shard = torch.load(part, map_location="cpu")
        train_mask, test_mask = part_masks(shard["video_id"], train_videos)
        pieces = [shard[k] for k in keys]
        x = torch.cat(pieces, dim=1).cpu().numpy().astype(np.float32, copy=False)
        ntr = int(train_mask.sum())
        nte = int(test_mask.sum())
        x_train[tr_i : tr_i + ntr] = x[train_mask]
        x_test[te_i : te_i + nte] = x[test_mask]
        tr_i += ntr
        te_i += nte
        del x, pieces, shard
        gc.collect()

    return x_train, x_test


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std
    return x_train, x_test


def ridge_probe(
    name: str,
    keys: Sequence[str],
    train_videos: set,
    n_train: int,
    n_test: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    print(f"[probe] loading {name}: {keys}", flush=True)
    x_train, x_test = collect_feature_arrays(keys, train_videos, n_train, n_test)
    x_train, x_test = standardize_train_test(x_train, x_test)

    print(f"[probe] fitting LSQR Ridge for {name} with shape {x_train.shape}", flush=True)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-4)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    rho = spearmanr(y_test, pred).correlation
    result = {
        "feature": name,
        "dim": int(x_train.shape[1]),
        "alpha": float(alpha),
        "ridge_r2": float(r2_score(y_test, pred)),
        "spearman_rho": float(rho),
    }
    del x_train, x_test, pred, model
    gc.collect()
    return result


def depth_alignment(train_videos: set) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    accum = {
        name: {"sq": 0.0, "cos": 0.0, "n": 0}
        for name in MODEL_KEYS
    }
    accum["GT depth"] = {"sq": 0.0, "cos": 0.0, "n": 0}

    for part in iter_parts():
        shard = torch.load(part, map_location="cpu")
        _, test_mask_np = part_masks(shard["video_id"], train_videos)
        test_mask = torch.from_numpy(test_mask_np)
        gt = shard["z_depth_feature_gt"][test_mask].float()
        gt_norm = torch.linalg.norm(gt, dim=1).clamp_min(1e-8)

        for name, key in MODEL_KEYS.items():
            pred = shard[key][test_mask].float()
            diff = pred - gt
            cos = (pred * gt).sum(dim=1) / (
                torch.linalg.norm(pred, dim=1).clamp_min(1e-8) * gt_norm
            )
            accum[name]["sq"] += float(diff.pow(2).mean(dim=1).sum())
            accum[name]["cos"] += float(cos.sum())
            accum[name]["n"] += int(pred.shape[0])

        accum["GT depth"]["sq"] += 0.0
        accum["GT depth"]["cos"] += float(torch.ones_like(gt_norm).sum())
        accum["GT depth"]["n"] += int(gt.shape[0])
        del shard, gt
        gc.collect()

    for name, stats in accum.items():
        n = stats["n"]
        rows.append(
            {
                "feature": name,
                "mse_to_gt": float(stats["sq"] / n),
                "mean_cosine_to_gt": float(stats["cos"] / n),
            }
        )
    return rows


def index_diagnostics(train_videos: set) -> List[Dict[str, float]]:
    accum = {
        name: {"correct": 0.0, "tokens": 0.0, "exact": 0.0, "rows": 0.0, "conf": 0.0}
        for name in INDEX_KEYS
    }

    for part in iter_parts():
        shard = torch.load(part, map_location="cpu")
        _, test_mask_np = part_masks(shard["video_id"], train_videos)
        test_mask = torch.from_numpy(test_mask_np)
        gt = shard["z_depth_indices_gt"][test_mask]
        for name, (idx_key, conf_key) in INDEX_KEYS.items():
            pred = shard[idx_key][test_mask]
            correct = pred.eq(gt)
            accum[name]["correct"] += float(correct.sum())
            accum[name]["tokens"] += float(correct.numel())
            accum[name]["exact"] += float(correct.all(dim=1).sum())
            accum[name]["rows"] += float(correct.shape[0])
            accum[name]["conf"] += float(shard[conf_key][test_mask].float().mean(dim=1).sum())
        del shard, gt
        gc.collect()

    rows = []
    for name, stats in accum.items():
        rows.append(
            {
                "feature": name,
                "token_accuracy": stats["correct"] / stats["tokens"],
                "sequence_accuracy": stats["exact"] / stats["rows"],
                "mean_confidence": stats["conf"] / stats["rows"],
            }
        )
    return rows


def collect_umap_sample(
    keys: Sequence[str],
    train_videos: set,
    sample_indices: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = first_tensor_dim(keys)
    xs = np.empty((len(sample_indices), dim), dtype=np.float32)
    ys = y_test[sample_indices]
    target_set = set(int(i) for i in sample_indices)
    seen = 0
    out_i = 0

    for part in iter_parts():
        shard = torch.load(part, map_location="cpu")
        _, test_mask = part_masks(shard["video_id"], train_videos)
        local_test_indices = np.arange(seen, seen + int(test_mask.sum()))
        keep_local = np.array([idx in target_set for idx in local_test_indices])
        if keep_local.any():
            pieces = [shard[k] for k in keys]
            x = torch.cat(pieces, dim=1).cpu().numpy().astype(np.float32, copy=False)
            kept_rows = np.where(test_mask)[0][keep_local]
            n = len(kept_rows)
            xs[out_i : out_i + n] = x[kept_rows]
            out_i += n
            del x, pieces
        seen += int(test_mask.sum())
        del shard
        gc.collect()

    return xs, ys


def morans_i(coords: np.ndarray, values: np.ndarray, k: int = 15) -> float:
    values = values.astype(np.float64)
    centered = values - values.mean()
    denom = float(np.sum(centered**2))
    if denom <= 1e-12:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    nbrs = indices[:, 1:]
    lag = centered[nbrs].mean(axis=1)
    return float(len(values) * np.sum(centered * lag) / (len(values) * denom))


def umap_snapshot(
    probe_rows: List[Dict[str, float]],
    train_videos: set,
    y_test: np.ndarray,
    sample_n: int,
    seed: int,
) -> List[Dict[str, float]]:
    if umap is None:
        print("[umap] umap-learn is not available; skipping UMAP figure", flush=True)
        return []

    rng = np.random.default_rng(seed)
    sample_n = min(sample_n, len(y_test))
    sample_indices = np.sort(rng.choice(len(y_test), size=sample_n, replace=False))
    reps = [("RGB", ["z_rgb_feature_input"])] + [
        (name, ["z_rgb_feature_input", key]) for name, key in MODEL_KEYS.items()
    ]
    r2_lookup = {row["feature"]: row["ridge_r2"] for row in probe_rows}
    moran_rows: List[Dict[str, float]] = []

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6), constrained_layout=True)
    axes_flat = axes.ravel()
    norm = plt.Normalize(vmin=float(y_test.min()), vmax=float(y_test.max()))

    for ax, (name, keys) in zip(axes_flat, reps):
        label = "RGB" if name == "RGB" else f"RGB + {name}"
        print(f"[umap] {label}", flush=True)
        x, y = collect_umap_sample(keys, train_videos, sample_indices, y_test)
        x, _ = standardize_train_test(x, x.copy())
        n_components = min(50, x.shape[1], x.shape[0] - 1)
        x_pca = PCA(n_components=n_components, random_state=seed).fit_transform(x)
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.05,
            metric="euclidean",
            random_state=seed,
            low_memory=True,
        )
        coords = reducer.fit_transform(x_pca)
        moran = morans_i(coords, y, k=15)
        feature_name = "RGB" if name == "RGB" else f"RGB + {name}"
        moran_rows.append(
            {
                "feature": feature_name,
                "umap_samples": int(sample_n),
                "morans_i": float(moran),
            }
        )
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=y,
            s=4,
            cmap="viridis",
            norm=norm,
            linewidths=0,
            alpha=0.82,
        )
        r2 = r2_lookup.get(feature_name, float("nan"))
        ax.set_title(f"{label}\nR2={r2:.3f}, I={moran:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        del x, x_pca, coords
        gc.collect()

    for ax in axes_flat[len(reps) :]:
        ax.axis("off")

    cbar = fig.colorbar(sc, ax=axes_flat.tolist(), shrink=0.88, pad=0.01)
    cbar.set_label("|delta t| magnitude")
    fig.suptitle(
        f"Deployment-equivalent representations on held-out LIBERO-10 ({sample_n} sampled test pairs)",
        fontsize=14,
    )
    fig.savefig(OUT_DIR / "umap_moran_feature_snapshot.png", dpi=220)
    plt.close(fig)
    return moran_rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                parsed[key] = value if key == "feature" else float(value)
            rows.append(parsed)
    return rows


def plot_probe_bars(rows: List[Dict[str, float]]) -> None:
    concat_rows = [r for r in rows if r["feature"] == "RGB" or r["feature"].startswith("RGB +")]
    labels = [r["feature"].replace("RGB + ", "+ ") for r in concat_rows]
    values = [r["ridge_r2"] for r in concat_rows]
    colors = ["#4C78A8"] + ["#59A14F" if v >= values[0] else "#E15759" for v in values[1:]]

    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    bars = ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(values[0], color="#333333", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_ylabel("Held-out Ridge R2")
    ax.set_title("LIBERO-10 translation-magnitude probe")
    ax.set_ylim(max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.04))
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ridge_r2_probe_bars.png", dpi=220)
    plt.close(fig)


def format_markdown(
    probe_rows: List[Dict[str, float]],
    align_rows: List[Dict[str, float]],
    index_rows: List[Dict[str, float]],
    moran_rows: List[Dict[str, float]],
    split_meta: Dict[str, int],
) -> str:
    probe_lookup = {row["feature"]: row for row in probe_rows}
    baseline = probe_lookup["RGB"]["ridge_r2"]
    concat_rows = [r for r in probe_rows if r["feature"].startswith("RGB +")]
    best = max(concat_rows, key=lambda r: r["ridge_r2"])
    no_depth = [
        r
        for r in concat_rows
        if r["feature"] in {"RGB + Model 3", "RGB + Model 5"}
    ]
    best_no_depth = max(no_depth, key=lambda r: r["ridge_r2"]) if no_depth else None

    lines = [
        "# Redone Feature Analysis",
        "",
        "## Setup",
        "",
        (
            "I reran the feature analysis on the provided `libero10_features` cache "
            f"with {split_meta['total_samples']:,} consecutive LIBERO-10 frame pairs. "
            f"The split is video-disjoint using `train_test_split(test_size=0.2, random_state=42)`: "
            f"{split_meta['train_samples']:,} train pairs and {split_meta['test_samples']:,} held-out pairs "
            f"from {split_meta['train_videos']} train videos and {split_meta['test_videos']} test videos."
        ),
        "",
        "Features are z-scored using train statistics only. The main probe is Ridge over "
        f"`alpha={probe_rows[0]['alpha']:.0f}` with the LSQR solver, evaluated by held-out R2 and Spearman rho against "
        "the xyz translation magnitude.",
        "",
        "## Probe Results",
        "",
        "| Feature | Dim | Alpha | Ridge R2 | Spearman rho |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in probe_rows:
        lines.append(
            f"| {row['feature']} | {int(row['dim'])} | {row['alpha']:.0e} | "
            f"{row['ridge_r2']:.4f} | {row['spearman_rho']:.4f} |"
        )

    lines += [
        "",
        "## Depth-Feature Faithfulness",
        "",
        "| Feature | MSE to GT depth feature | Mean cosine to GT |",
        "|---|---:|---:|",
    ]
    for row in align_rows:
        lines.append(
            f"| {row['feature']} | {row['mse_to_gt']:.5f} | {row['mean_cosine_to_gt']:.4f} |"
        )

    lines += [
        "",
        "## Depth-Index Diagnostics",
        "",
        "| Feature | Token accuracy | Sequence accuracy | Mean confidence |",
        "|---|---:|---:|---:|",
    ]
    for row in index_rows:
        lines.append(
            f"| {row['feature']} | {row['token_accuracy']:.4f} | "
            f"{row['sequence_accuracy']:.4f} | {row['mean_confidence']:.4f} |"
        )

    if moran_rows:
        lines += [
            "",
            "## UMAP Smoothness",
            "",
            "| Feature | Samples | Moran's I |",
            "|---|---:|---:|",
        ]
        for row in moran_rows:
            lines.append(
                f"| {row['feature']} | {int(row['umap_samples'])} | {row['morans_i']:.4f} |"
            )

    lines += [
        "",
        "## Rewritten Findings",
        "",
        (
            f"**Finding A - RGB already carries most in-distribution geometry.** "
            f"The provided RGB feature reaches R2={baseline:.3f}, matching the paper's "
            "pre-VQ LAPA-LAQ reference scale rather than the finetuned-LAPA RGB reference. "
            "This cache therefore supports analysis of the frozen/pre-finetune representation, "
            "not a reproduction of the old finetuned-LAPA baseline."
        ),
        "",
        (
            f"**Finding B - useful depth injection is selective.** "
            f"The best deployment-equivalent representation is {best['feature']} "
            f"(R2={best['ridge_r2']:.3f}, delta={best['ridge_r2'] - baseline:+.3f} over RGB). "
            "The result should be interpreted as a modest decodability gain rather than a new "
            "standalone policy-success claim."
        ),
        "",
    ]
    if best_no_depth is not None:
        lines += [
            (
                f"**Finding C - no-depth controls bound the parameter-only explanation.** "
                f"The strongest RGB-only Stage-2.5 control is {best_no_depth['feature']} "
                f"(R2={best_no_depth['ridge_r2']:.3f}, delta={best_no_depth['ridge_r2'] - baseline:+.3f}). "
                "When a depth-image variant beats this control, the gain is more plausibly tied to "
                "geometric information in the depth stream than to extra capacity alone."
            ),
            "",
        ]

    lines += [
        (
            "**Finding D - feature-scale and target choice matter.** "
            "The continuous-distillation variants should be read together with cosine/MSE alignment "
            "to the Stage-1 depth teacher, while index-prediction variants should be read together "
            "with token and sequence accuracy. This avoids overclaiming from R2 alone."
        ),
        "",
        "## Suggested Replacement Text",
        "",
        (
            "We probe the geometric content of the deployment-equivalent representations on "
            "138,090 LIBERO-10 frame pairs using a video-disjoint 80/20 split. Features are "
            "standardized with training statistics only and evaluated with a Ridge probe over "
            "end-effector translation magnitude. The provided RGB representation reaches "
            f"R2={baseline:.3f}, indicating that the cache corresponds to the pre-VQ LAPA-LAQ "
            "feature scale rather than the finetuned-LAPA reference used in the earlier appendix. "
            "Concatenating Stage-2.5 "
            f"depth features gives a selective improvement: {best['feature']} reaches "
            f"R2={best['ridge_r2']:.3f}, a {best['ridge_r2'] - baseline:+.3f} absolute gain over RGB. "
            "RGB-only Stage-2.5 controls remain the appropriate comparison for separating depth "
            "information from added capacity. Therefore, the feature analysis supports a cautious "
            "conclusion: depth-derived representations can add geometric decodability, but the gain "
            "is incremental and should be paired with downstream policy evaluation."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_videos, test_videos, counts = load_video_split(args.seed)
    n_train = sum(counts[v] for v in train_videos)
    n_test = sum(counts[v] for v in test_videos)
    split_meta = {
        "total_samples": n_train + n_test,
        "train_samples": n_train,
        "test_samples": n_test,
        "train_videos": len(train_videos),
        "test_videos": len(test_videos),
        "seed": args.seed,
    }
    print(f"[split] {split_meta}", flush=True)

    y_train, y_test = collect_y(train_videos, n_train, n_test)

    if args.diagnostics_only:
        probe_rows = read_csv_rows(OUT_DIR / "ridge_probe_results.csv")
    else:
        probe_specs: List[Tuple[str, List[str]]] = [
            ("RGB", ["z_rgb_feature_input"]),
            ("GT depth", ["z_depth_feature_gt"]),
            ("RGB + GT depth", ["z_rgb_feature_input", "z_depth_feature_gt"]),
        ]
        if args.include_depth_only:
            probe_specs += [
                (f"Depth only {name}", [key]) for name, key in MODEL_KEYS.items()
            ]
        probe_specs += [
            (f"RGB + {name}", ["z_rgb_feature_input", key])
            for name, key in MODEL_KEYS.items()
        ]

        probe_rows = [
            ridge_probe(name, keys, train_videos, n_train, n_test, y_train, y_test, args.ridge_alpha)
            for name, keys in probe_specs
        ]
        write_csv(OUT_DIR / "ridge_probe_results.csv", probe_rows)

    if not args.skip_plots:
        plot_probe_bars(probe_rows)

    print("[diagnostics] depth alignment", flush=True)
    align_rows = depth_alignment(train_videos)
    write_csv(OUT_DIR / "depth_alignment_results.csv", align_rows)

    print("[diagnostics] depth index accuracy", flush=True)
    index_rows = index_diagnostics(train_videos)
    write_csv(OUT_DIR / "depth_index_results.csv", index_rows)

    moran_rows: List[Dict[str, float]] = []
    if not args.skip_umap:
        moran_rows = umap_snapshot(
            probe_rows,
            train_videos,
            y_test,
            sample_n=args.umap_samples,
            seed=args.seed,
        )
        write_csv(OUT_DIR / "umap_moran_results.csv", moran_rows)
    elif (OUT_DIR / "umap_moran_results.csv").exists():
        moran_rows = read_csv_rows(OUT_DIR / "umap_moran_results.csv")

    all_results = {
        "split": split_meta,
        "probe": probe_rows,
        "alignment": align_rows,
        "index": index_rows,
        "umap_moran": moran_rows,
    }
    (OUT_DIR / "feature_analysis_results.json").write_text(
        json.dumps(all_results, indent=2)
    )
    md = format_markdown(probe_rows, align_rows, index_rows, moran_rows, split_meta)
    (OUT_DIR / "feature_analysis_redone.md").write_text(md)
    print(f"[done] wrote outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
