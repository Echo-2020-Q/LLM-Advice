"""Read-only validation and compact, hash-verified caches for the Qwen scan.

Run with the visualization environment, for example::

    python tools/qwen_scan_data.py
    python tools/qwen_scan_data.py --use-cache

No simulator or API is invoked. Raw results are never modified. A cache is
reused only after every source file and every cached artifact passes SHA256.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from io import BytesIO
from itertools import product
import json
from pathlib import Path, PureWindowsPath
import platform

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "main/SPD/results/scan_20261027_122530_2次Qwen微扫描"
DEFAULT_REPORT = ROOT / "reports/qwen_20261027_repeats2"
CACHE_VERSION = 1
KEY_COLUMNS = ["b", "lambda_prob", "llm_access_fraction", "seed"]
GROUPS = ("population", "eligible", "ineligible")
MEAN_METRICS = (
    "cumulative_mean_payoff", "time_mean_payoff", "early_mean_payoff",
    "tail_mean_payoff",
)


def _require(ok, label):
    if not bool(ok):
        raise ValueError("Data audit failed: " + label)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _source_name(path):
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _source_path(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def _read(path, hashes):
    raw = path.read_bytes()
    hashes[_source_name(path)] = _sha(raw)
    return raw


def _json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


class Checks:
    """Aggregate successful comparisons without retaining individual histories."""

    def __init__(self):
        self.data = {}

    def require(self, ok, category, label=""):
        _require(ok, category + (": " + label if label else ""))
        entry = self.data.setdefault(category, {"passed": True, "comparisons": 0})
        entry["comparisons"] += 1

    def close(self, a, b, category, label=""):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        self.require(np.allclose(a, b, rtol=1e-10, atol=1e-8, equal_nan=True),
                     category, label)
        diff = np.abs(a - b)
        finite = diff[np.isfinite(diff)]
        max_error = float(np.max(finite)) if finite.size else 0.0
        entry = self.data[category]
        entry["max_absolute_error"] = max(entry.get("max_absolute_error", 0), max_error)

    def merge(self, other):
        for name, item in other.items():
            entry = self.data.setdefault(name, {"passed": True, "comparisons": 0})
            entry["comparisons"] += item["comparisons"]
            if "max_absolute_error" in item:
                entry["max_absolute_error"] = max(entry.get("max_absolute_error", 0),
                                                   item["max_absolute_error"])


def _key(row):
    return (float(row["b"]), float(row["lambda_prob"]),
            float(row["llm_access_fraction"]), int(row["seed"]))


def _audit_one(args):
    batch, row, config = args
    key = _key(row)
    label = str(key)
    checks, hashes = Checks(), {}
    run = (batch / row["output_dir"]).resolve()
    checks.require(run.is_relative_to(batch), "output_paths_within_batch", label)
    h = pd.read_csv(BytesIO(_read(run / "history.csv", hashes)))
    raw_grid = pd.read_csv(BytesIO(_read(run / "final_grid.csv", hashes)),
                           header=None).to_numpy()
    steps = int(config["steps"])
    early = int(config["early_average_steps"])
    tail = int(config["tail_average_steps"])
    tail_start = steps - tail + 1
    checks.require(len(h) == steps + 1, "complete_history_length", label)
    checks.close(h["round"], np.arange(steps + 1), "complete_round_sequence", label)
    checks.require(raw_grid.shape == (50, 50) and set(raw_grid.ravel()) <= {"C", "D"},
                   "final_grid_shape_and_actions", label)
    grid = raw_grid == "C"
    checks.close(grid.mean(), row["final_cooperation_rate"], "final_grid_cooperation", label)
    ints = grid.astype(np.int8)
    neighbors = (np.roll(ints, 1, 0) + np.roll(ints, -1, 0)
                 + np.roll(ints, 1, 1) + np.roll(ints, -1, 1))
    final_payoffs = neighbors * np.where(grid, 1.0, float(row["b"]))
    checks.close(final_payoffs.mean(), row["final_average_payoff"],
                 "final_grid_recomputed_payoff", label)
    n = int(row["population_count"])
    checks.require(n == 2500, "population_size", label)
    checks.require(int(row["eligible_count"]) + int(row["ineligible_count"]) == n,
                   "group_counts", label)
    checks.close(row["actual_llm_access_fraction"], row["eligible_count"] / n,
                 "actual_eligible_fraction", label)
    checks.close(row["llm_access_fraction"], row["actual_llm_access_fraction"],
                 "requested_eligible_fraction", label)
    checks.close(row["llm_access_count"], row["eligible_count"], "group_counts", label)

    for prefix in ("", "control_"):
        coop = h[prefix + "cooperation_rate"].to_numpy(dtype=float)
        checks.require(np.isfinite(coop).all() and ((coop >= 0) & (coop <= 1)).all(),
                       "valid_cooperation_rates", label + prefix)
        checks.close(coop * n, np.rint(coop * n), "integer_cooperator_counts", label)
        checks.close(coop[0], row["initial_cooperation_rate"], "paired_initial_cooperation", label)
        checks.close(coop[-1], row[prefix + "final_cooperation_rate"],
                     "summary_final_cooperation", label)
        checks.close(coop[tail_start:].mean(), row[prefix + "tail_mean_cooperation_rate"],
                     "summary_tail_cooperation", label)
        checks.close(coop[1:].mean(), row[prefix + "time_mean_cooperation_rate"],
                     "summary_time_cooperation", label)
        checks.close(h[prefix + "average_payoff"], h[prefix + "population_mean_payoff"],
                     "population_payoff_alias", label)
        checks.close(h[prefix + "average_payoff"].iloc[-1],
                     row[prefix + "final_average_payoff"], "summary_final_payoff", label)
        checks.close(h[prefix + "average_payoff"].iloc[tail_start:].mean(),
                     row[prefix + "tail_mean_payoff"], "summary_tail_payoff_alias", label)
        zero = coop == 0
        zero_indices = np.flatnonzero(zero)
        entries = int(zero[0]) + int(np.sum(~zero[:-1] & zero[1:]))
        revivals = int(np.sum(zero[:-1] & ~zero[1:]))
        final_start = np.nan
        if zero[-1]:
            nonzero_indices = np.flatnonzero(~zero)
            final_start = int(nonzero_indices[-1] + 1) if len(nonzero_indices) else 0
        extinction_values = {
            "first_extinction_round": int(zero_indices[0]) if len(zero_indices) else np.nan,
            "extinction_observed": bool(zero.any()), "extinction_events": entries,
            "cooperation_revivals": revivals, "extinct_at_end": bool(zero[-1]),
            "final_extinction_start_round": final_start,
        }
        for name, expected in extinction_values.items():
            checks.close(row[prefix + name], expected, "extinction_history_semantics", label + name)
        for name, expected in (("observed_steps", steps), ("early_observed_steps", early),
                               ("tail_observed_steps", tail)):
            checks.close(row[prefix + name], expected, "observed_window_lengths", label)

        weighted = np.zeros(steps + 1)
        for group in GROUPS:
            count = int(row[prefix + group + "_count"])
            checks.close(count, row[group + "_count"], "paired_group_counts", label)
            vals = h[prefix + group + "_mean_payoff"].to_numpy(dtype=float)
            cumulative = h[prefix + group + "_cumulative_mean_payoff"].to_numpy(dtype=float)
            if count == 0:
                checks.require(np.isnan(vals).all() and np.isnan(cumulative).all(),
                               "empty_group_history_means_nan", label + prefix + group)
                checks.require(all(pd.isna(row[prefix + group + "_" + metric])
                                   for metric in MEAN_METRICS),
                               "empty_group_summary_means_nan", label + prefix + group)
                checks.close(row[prefix + group + "_cumulative_total_payoff"], 0,
                             "empty_group_total_payoff_zero", label)
                continue
            checks.require(np.isfinite(vals).all() and (vals >= 0).all(),
                           "finite_nonnegative_group_payoffs", label + prefix + group)
            checks.close(cumulative, np.r_[0., np.cumsum(vals[1:])],
                         "history_cumulative_excludes_round_zero", label + prefix + group)
            expected = {
                "cumulative_mean_payoff": vals[1:].sum(),
                "time_mean_payoff": vals[1:].mean(),
                "early_mean_payoff": vals[1:early + 1].mean(),
                "tail_mean_payoff": vals[tail_start:].mean(),
                "cumulative_total_payoff": vals[1:].sum() * count,
            }
            for metric, value in expected.items():
                checks.close(row[prefix + group + "_" + metric], value,
                             "summary_" + metric, label + prefix + group)
            if group != "population":
                weighted += (count / n) * vals
        checks.close(weighted, h[prefix + "population_mean_payoff"],
                     "population_weighting_nonempty_groups_only", label + prefix)

    for col in (col for col in row if col.startswith("delta_")):
        base = col.removeprefix("delta_")
        checks.close(row[col], row[base] - row["control_" + base],
                     "summary_paired_delta", label + col)
    for col in (col for col in h if col.startswith("delta_")):
        base = col.removeprefix("delta_")
        checks.close(h[col], h[base] - h["control_" + base],
                     "history_paired_delta", label + col)
    if row["lambda_prob"] == 0 or row["llm_access_fraction"] == 0:
        for col in (col for col in h if col.startswith("delta_")):
            values = h[col].to_numpy(dtype=float)
            checks.close(values[np.isfinite(values)], 0, "zero_intervention_history_delta", label)
        delta_values = np.asarray([row[col] for col in row if col.startswith("delta_")], float)
        checks.close(delta_values[np.isfinite(delta_values)], 0,
                     "zero_intervention_summary_delta", label)
    for counter in ("advice_requests", "advice_used", "fermi_updates", "strategy_changes"):
        vals = h[counter].to_numpy()
        checks.require(np.isfinite(vals).all() and (vals >= 0).all(),
                       "valid_update_counters", label + counter)
        checks.close(vals[0], 0, "initial_update_counters_zero", label)
        checks.close(vals[1:].sum(), row["total_" + counter], "summary_update_counters", label)
    checks.close(h["advice_used"].iloc[1:] + h["fermi_updates"].iloc[1:], n,
                 "one_update_per_player_per_round", label)
    checks.close(h["advice_requests"], h["advice_used"], "complete_advice_coverage", label)

    coop = h["cooperation_rate"].to_numpy(dtype=float).copy()
    prev_start = max(1, tail_start - tail)
    prev_mean = float(coop[prev_start:tail_start].mean())
    extras = {
        "cooperation_early_mean": float(coop[1:early + 1].mean()),
        "cooperation_prev_tail_mean": prev_mean,
        "cooperation_tail_drift": float(coop[tail_start:].mean() - prev_mean),
        "cooperation_3001_4000_mean": float(coop[3001:4001].mean()),
        "cooperation_4001_5000_mean": float(coop[4001:5001].mean()),
        "cooperation_tail_min": float(coop[tail_start:].min()),
        "cooperation_tail_max": float(coop[tail_start:].max()),
        "cooperation_tail_sd_time": float(coop[tail_start:].std(ddof=1)),
        "cooperation_last_100_mean": float(coop[-100:].mean()),
        "first_all_cooperate_round": (int(np.flatnonzero(coop == 1)[0])
                                       if np.any(coop == 1) else np.nan),
    }
    baseline_columns = ["control_cooperation_rate", "control_average_payoff",
                        "control_population_mean_payoff", "control_population_cumulative_mean_payoff"]
    baseline_hash = _sha(np.asarray(h[baseline_columns].to_numpy(), dtype="<f8").tobytes())
    return key, coop, grid, extras, baseline_hash, hashes, checks.data


def _load_advisor(config, hashes, checks):
    sources = config["advisor_sources"]
    checks.require(bool(sources), "advisor_sources_present")
    source = sources[0]
    fingerprint = {name: item["sha256"] for name, item in source["source_files"].items()}
    checks.require(all({name: v["sha256"] for name, v in s["source_files"].items()} == fingerprint
                       for s in sources), "single_advisor_source_fingerprint")
    checks.require(all(s["model"] == source["model"] and s["batch_id"] == source["batch_id"]
                       for s in sources), "single_advisor_model_and_batch")
    directory = ROOT / "logs/GPT_Multistepsa" / PureWindowsPath(config["advisor_batch"]).name
    raw_files = {}
    for name, expected in fingerprint.items():
        raw_files[name] = _read(directory / name, hashes)
        checks.require(_sha(raw_files[name]) == expected, "advisor_recorded_sha256", name)
    cfg = json.loads(raw_files["config.json"])
    summary = json.loads(_read(directory / "summary.json", hashes))
    advisor = pd.read_csv(BytesIO(raw_files["results.csv"]))
    checks.require(_sha(cfg["system_prompt"].encode()) == source["system_prompt_sha256"],
                   "advisor_prompt_sha256")
    checks.require(cfg["model"] == source["model"] == summary["model"], "advisor_model_metadata")
    checks.require(summary["status"] == "completed" and summary["successful_requests"] == 1024
                   and summary["failed_requests"] == 0, "advisor_batch_completed")
    checks.require(len(advisor) == 1024 and (advisor["status"] == "success").all(),
                   "advisor_1024_successful_rows")
    checks.require(set(advisor.grid_key) == set(map("".join, product("CD", repeat=9))),
                   "advisor_all_512_states")
    checks.require((advisor.groupby("grid_key").size() == 2).all(), "advisor_two_answers_per_state")
    checks.require(set(advisor.recommendation) <= {"C", "D"}, "advisor_valid_actions")
    advisor["center"] = advisor.grid_key.str[4]
    advisor["cardinal_c"] = advisor.grid_key.map(lambda key: sum(key[i] == "C" for i in (1, 3, 5, 7)))
    advisor["is_c"] = advisor.recommendation.eq("C")
    checks.require((advisor.center == advisor.current_action).all()
                   and (advisor.cardinal_c == advisor.cooperating_neighbors).all(),
                   "advisor_grid_metadata")
    checks.require(cfg["b"] is None and cfg["b_constraint"] == "1 < b < 2",
                   "advisor_unknown_b_shared_policy")
    counts = advisor.groupby("grid_key")["is_c"].sum()
    metadata = {
        "directory": _source_name(directory), "model": source["model"],
        "batch_id": source["batch_id"], "model_identity_basis": "local recorded metadata",
        "states": 512, "answers_per_state": 2, "successful_answers": 1024,
        "cooperate_answers": int(advisor.is_c.sum()),
        "uniform_state_cooperation_probability": float(advisor.is_c.mean()),
        "agreement_states": int(counts.isin([0, 2]).sum()),
        "agreement_fraction": float(counts.isin([0, 2]).mean()),
        "answer_pairs": {"CC": int((counts == 2).sum()), "CD": int((counts == 1).sum()),
                         "DD": int((counts == 0).sum())},
        "all_C_answers": advisor.loc[advisor.grid_key == "C" * 9, "recommendation"].tolist(),
        "all_D_answers": advisor.loc[advisor.grid_key == "D" * 9, "recommendation"].tolist(),
        "conditioning": "same table for every b; b unknown except 1 < b < 2; numerical K/P/lambda absent",
        "sampling": source["sampling"],
        "probability_caveat": "Uniformly weighted enumerated states, not visited-state weighting in simulation.",
    }
    return advisor, metadata


def _verify_cached_sources(metadata):
    pairs = list(metadata["source_sha256"].items())

    def check_one(pair):
        name, expected = pair
        path = _source_path(name)
        _require(path.is_file() and _sha(path.read_bytes()) == expected,
                 "cache source SHA256 changed or missing: " + name)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(check_one, pairs), 1):
            if index % 100 == 0 or index == len(pairs):
                print(f"Verified cache source hashes {index}/{len(pairs)}", flush=True)


def _try_cache(batch, report):
    metadata_path = report / "data/cache_metadata.json"
    if not metadata_path.exists():
        print("No cache found; performing full source audit.", flush=True)
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_version") != CACHE_VERSION or metadata.get("module_sha256") != _sha(Path(__file__).read_bytes()):
        print("Cache implementation changed; performing full source audit.", flush=True)
        return None
    _require(metadata["batch"] == _source_name(batch), "cache belongs to another batch")
    _verify_cached_sources(metadata)
    for name, expected in metadata["artifact_sha256"].items():
        path = report / name
        _require(path.is_file() and _sha(path.read_bytes()) == expected,
                 "cached artifact changed or missing: " + name)
    config = json.loads((batch / "scan_config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(report / "data/validated_runs.csv")
    advisor = pd.read_csv(report / "data/advisor_answers.csv")
    audit = json.loads((report / "数据核验.json").read_text(encoding="utf-8"))
    keys = [tuple([float(k[0]), float(k[1]), float(k[2]), int(k[3])]) for k in metadata["ordered_keys"]]
    with np.load(report / "data/trajectory_cache.npz", allow_pickle=False) as arrays:
        cooperation = arrays["cooperation"]
        final_grids = arrays["final_grids"]
    trajectories = {key: cooperation[i].copy() for i, key in enumerate(keys)}
    grids = {key: final_grids[i].copy() for i, key in enumerate(keys)}
    _require(len(df) == len(keys) == len(trajectories) == 616, "cached run count")
    print("Source hashes verified; reused audited trajectories and grids.", flush=True)
    return config, df, trajectories, grids, advisor, audit


def load_scan(batch: Path, report: Path, use_cache: bool = False):
    """Return config, validated rows, cooperation arrays, C=True grids, advice, audit.

    Dictionary keys are (float b, float lambda, float P, int seed). Each
    cooperation array contains round 0 through round 10000. Original summary
    columns, including NaN empty-group means, are preserved in the returned df.
    """
    batch, report = Path(batch).resolve(), Path(report).absolute()
    # WSL resolves this repository to its older mixed-case spelling, while the
    # current sandbox grants writes through the lowercase spelling. Preserve a
    # verified alias for output; source identity and hashes still use ROOT.
    lowercase_root = Path(str(ROOT).lower())
    if (lowercase_root.is_dir() and lowercase_root.samefile(ROOT)
            and report.is_relative_to(ROOT)):
        report = lowercase_root / report.relative_to(ROOT)
    (report / "data").mkdir(parents=True, exist_ok=True)
    if use_cache:
        cached = _try_cache(batch, report)
        if cached is not None:
            return cached
    checks, hashes = Checks(), {}
    config = json.loads(_read(batch / "scan_config.json", hashes))
    df = pd.read_csv(BytesIO(_read(batch / "summary.csv", hashes))).sort_values("run_id").reset_index(drop=True)
    checks.require(config["status"] == "completed" and (df.status == "completed").all(),
                   "all_runs_completed")
    checks.require(len(df) == config["total_runs"] == config["completed_runs"] == 616,
                   "complete_616_runs")
    checks.require(config["parameter_sets"] == 308 and config["repeats"] == 2,
                   "design_308_sets_two_repeats")
    checks.require(config["seed_values"] == [42, 43], "two_declared_seeds")
    expected = set(product(config["scan_values"]["b"], config["scan_values"]["lambda_prob"],
                           config["scan_values"]["llm_access_fraction"], config["seed_values"]))
    actual = {_key(row) for row in df.to_dict("records")}
    checks.require(len(actual) == len(df) and actual == expected, "complete_parameter_seed_product")
    checks.require(set(df.run_id) == set(config["completed_run_ids"]) == set(range(1, 617)),
                   "complete_run_ids")
    checks.require((df.groupby(KEY_COLUMNS[:-1]).seed.nunique() == 2).all(),
                   "two_unique_seeds_per_parameter_set")
    constants = {
        "k": 0.1, "size": 50, "steps": 10000, "repeats": 2,
        "boundary": "periodic", "update_mode": "synchronous", "initial_cooperation": 0.5,
        "tail_average_steps": 1000, "tail_start_round": 9001, "tail_end_round": 10000,
        "early_average_steps": 200, "early_start_round": 1, "early_end_round": 200,
        "payoff_tail_start_round": 9001, "payoff_tail_end_round": 10000,
        "payoff_windows_overlap": False, "paired_fermi_control": True,
    }
    for col, expected_value in constants.items():
        checks.require((df[col] == expected_value).all(), "fixed_parameters_and_windows", col)
    checks.require(config["steps"] == 10000 and config["early_average_steps"] == 200
                   and config["tail_average_steps"] == 1000 and config["paired_fermi_control"],
                   "scan_config_windows_and_pairing")
    checks.require(all(int(row.seed) == config["seed_values"][int(row.repeat_index) - 1]
                       for row in df.itertuples()), "repeat_index_to_seed_mapping")
    advisor, advisor_metadata = _load_advisor(config, hashes, checks)
    trajectories, grids, extra_rows, baseline_hashes = {}, {}, {}, {}
    rows = df.to_dict("records")
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = ((batch, row, config) for row in rows)
        for index, result in enumerate(pool.map(_audit_one, tasks), 1):
            key, coop, grid, extras, baseline_hash, new_hashes, new_checks = result
            trajectories[key], grids[key], extra_rows[key] = coop, grid, extras
            hashes.update(new_hashes)
            checks.merge(new_checks)
            control_key = (key[0], key[3])
            if control_key in baseline_hashes:
                checks.require(baseline_hashes[control_key] == baseline_hash,
                               "same_b_seed_control_population_trajectory_sha256", str(key))
            else:
                baseline_hashes[control_key] = baseline_hash
            if index % 100 == 0 or index == len(df):
                print(f"Audited {index}/{len(df)} runs", flush=True)
    for seed in config["seed_values"]:
        reference = trajectories[(config["scan_values"]["b"][0], 1.0, 1.0, seed)]
        for b in config["scan_values"]["b"]:
            checks.require(np.array_equal(reference, trajectories[b, 1.0, 1.0, seed]),
                           "all_advice_same_seed_trajectory_independent_of_b", str((b, seed)))
    extra_df = pd.DataFrame([extra_rows[_key(row)] for row in rows])
    df = pd.concat([df, extra_df], axis=1)
    audit = {
        "validation": "passed", "batch": _source_name(batch), "runs": len(df),
        "parameter_sets": 308, "repeats_per_parameter": 2, "seeds": [42, 43],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "windows": {"initial_round": 0, "payoff_full": [1, 10000], "early": [1, 200],
                    "tail": [9001, 10000], "previous_tail": [8001, 9000]},
        "checks": checks.data,
        "empty_groups": {
            "eligible_zero_population_runs": int((df.eligible_count == 0).sum()),
            "ineligible_zero_population_runs": int((df.ineligible_count == 0).sum()),
            "rule": "Empty-group per-person history/summary/delta values stay NaN; counts and total payoff are zero.",
            "weighting": "Population averages sum only nonempty group means times their population fractions; no zero-times-NaN.",
        },
        "statistical_unit": "One simulation seed. Paired intervention-minus-control differences precede aggregation.",
        "sample_size_limit": "Only seeds 42 and 43; two repeats cannot establish precise uncertainty or significance.",
        "same_advisor_table_limit": "All runs share one 512-state table with two LLM answers per state; simulation repeats do not replicate table generation.",
        "saved_data_limit": "Initial/final eligibility masks and initial grids were not saved. Initial cooperation rate and paired group counts are checked; exact cellwise initial equality cannot be independently reconstructed from these saved files.",
        "advisor": advisor_metadata,
        "control_population_hashes": [{"b": k[0], "seed": k[1], "sha256": v}
                                      for k, v in sorted(baseline_hashes.items())],
        "source_file_count": len(hashes), "source_sha256": dict(sorted(hashes.items())),
        "versions": {"python": platform.python_version(), "numpy": version("numpy"),
                     "pandas": version("pandas"), "cache_schema": CACHE_VERSION},
    }
    data_dir = report / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "validated_runs.csv", index=False, encoding="utf-8-sig")
    advisor.to_csv(data_dir / "advisor_answers.csv", index=False, encoding="utf-8-sig")
    keys = [_key(row) for row in rows]
    np.savez_compressed(data_dir / "trajectory_cache.npz",
                        cooperation=np.stack([trajectories[k] for k in keys]),
                        final_grids=np.stack([grids[k] for k in keys]),
                        run_ids=df.run_id.to_numpy(dtype=np.int32))
    _json_write(report / "数据核验.json", audit)
    artifact_names = ["data/validated_runs.csv", "data/advisor_answers.csv",
                      "data/trajectory_cache.npz", "数据核验.json"]
    metadata = {
        "cache_version": CACHE_VERSION, "module_sha256": _sha(Path(__file__).read_bytes()),
        "batch": _source_name(batch), "ordered_keys": [list(k) for k in keys],
        "source_sha256": dict(sorted(hashes.items())),
        "artifact_sha256": {name: _sha((report / name).read_bytes()) for name in artifact_names},
    }
    _json_write(data_dir / "cache_metadata.json", metadata)
    print(f"Validation passed. Saved audit and compact cache to {report}", flush=True)
    return config, df, trajectories, grids, advisor, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    load_scan(args.batch, args.report, args.use_cache)


if __name__ == "__main__":
    main()
