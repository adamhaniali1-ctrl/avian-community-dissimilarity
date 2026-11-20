from __future__ import annotations

import math

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import ks_2samp

from .final_evaluation_utils import ALL_METHODS, CONDITIONS, HOLDOUTS, PREDICTIVE_METHODS, RANKING_METHODS, SSI_COLUMNS, bootstrap_group, compression_label, group_seed, percentile, point_metrics, wasserstein_1d

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = ROOT / "results/final_evaluation"

STAGE_PATHS = (
    "config/final_evaluation.yaml", "scripts/final_evaluation_utils.py",
    "scripts/run_final_evaluation.py", "scripts/validate_final_evaluation.py",
    "audit_reports/final_evaluation_audit.md", "results/final_evaluation",
)

IDENTIFIERS = ["outer_holdout", "data_condition", "dataset", "pair_role", "pair_id", "event_uid_a", "event_uid_b"]

ALIGNED_COLUMNS = IDENTIFIERS + ["true_jaccard_distance", *ALL_METHODS]

METRIC_COLUMNS = ["outer_holdout", "data_condition", "pair_role", "dataset", "method", "metric", "estimate", "n_pairs", "missing_reason"]

BOOTSTRAP_COLUMNS = ["outer_holdout", "data_condition", "pair_role", "dataset", "method", "metric", "estimate", "ci_lower", "ci_upper", "n_pairs", "n_events", "requested_replicates", "valid_replicates", "undefined_replicates", "stream_seed", "stream_id", "missing_reason"]

def community_map(condition: str) -> dict[str, set[str]]:
    """Return exact event communities under the locked condition rule."""
    species = pd.read_csv(ROOT / "data_clean/event_species.csv", keep_default_na=False)
    communities = {event: set(group.species_name.astype(str)) for event, group in species.groupby("event_uid", sort=False)}
    all_events = pd.read_csv(ROOT / "data_clean/event_features.csv", usecols=["event_uid"]).event_uid.astype(str)
    output = {}
    for event in all_events:
        labels = set(communities.get(event, set()))
        if condition == "filtered": labels.discard("NA")
        output[event] = labels
    return output

def jaccard(a: set[str], b: set[str]) -> float:
    """Calculate the locked exact-set Jaccard distance."""
    union = a | b
    return 0.0 if not union else 1.0 - len(a & b) / len(union)

def align_condition(holdout: str, condition: str, communities: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Align all sealed and developmental predictions to independently verified targets."""
    folder = ROOT / f"experiments/holdout_{holdout}/{condition}"; manifest = pd.read_csv(folder / "outer_pairs.csv", keep_default_na=False, float_precision="round_trip")
    training = manifest[manifest.pair_role == "final_train"].copy(); evaluation = manifest[manifest.pair_role.isin(["familiar_test", "unseen_test"])].copy()
    if evaluation.pair_id.duplicated().any() or set(evaluation.pair_role) != {"familiar_test", "unseen_test"}: raise ValueError("Manifest evaluation rows invalid")
    recomputed = np.array([jaccard(communities[a], communities[b]) for a, b in zip(evaluation.event_uid_a, evaluation.event_uid_b)], dtype=np.float64)
    if not np.allclose(recomputed, evaluation.jaccard_distance.to_numpy(np.float64), rtol=0, atol=1e-15): raise ValueError("Manifest Jaccard reconstruction mismatch")
    baseline = pd.read_csv(ROOT / f"results/simple_baselines/holdout_{holdout}/{condition}/outer_predictions.csv", float_precision="round_trip")
    forest = pd.read_csv(ROOT / f"results/pairwise_random_forest/holdout_{holdout}/{condition}/outer_predictions.csv", float_precision="round_trip")
    sealed = ROOT / "results/final_fit_and_seal"; gdm = pd.read_csv(sealed / "gdm_predictions.csv", float_precision="round_trip"); gdm = gdm[(gdm.outer_holdout == holdout) & (gdm.data_condition == condition)]
    siamese = pd.read_csv(sealed / "siamese_ensemble_predictions.csv", float_precision="round_trip"); siamese = siamese[(siamese.outer_holdout == holdout) & (siamese.data_condition == condition)]
    base_columns = IDENTIFIERS + ["true_jaccard_distance", "training_mean_prediction", "unscaled_ssi_euclidean_distance", "calibrated_ssi_prediction"]
    forest_columns = IDENTIFIERS + ["true_jaccard_distance", "SSI_random_forest_prediction"]
    target = evaluation[IDENTIFIERS + ["jaccard_distance"]].copy()
    aligned = target.merge(baseline[base_columns], on=IDENTIFIERS, validate="one_to_one", how="outer", indicator="_base").merge(forest[forest_columns], on=IDENTIFIERS, validate="one_to_one", how="outer", indicator="_rf", suffixes=("", "_rf")).merge(gdm[IDENTIFIERS + ["prediction"]].rename(columns={"prediction": "gdm"}), on=IDENTIFIERS, validate="one_to_one", how="outer", indicator="_gdm").merge(siamese[IDENTIFIERS + ["ensemble_prediction"]].rename(columns={"ensemble_prediction": "siamese_ensemble_ssi"}), on=IDENTIFIERS, validate="one_to_one", how="outer", indicator="_siamese")
    if len(aligned) != len(evaluation) or any(not aligned[column].eq("both").all() for column in ["_base", "_rf", "_gdm", "_siamese"]): raise ValueError("Prediction pair coverage mismatch")
    target_values = aligned.jaccard_distance.to_numpy(np.float64)
    if not np.array_equal(target_values, aligned.true_jaccard_distance.to_numpy(np.float64)) or not np.array_equal(target_values, aligned.true_jaccard_distance_rf.to_numpy(np.float64)): raise ValueError("Saved target alignment mismatch")
    train_values = training.jaccard_distance.to_numpy(np.float64); mean = float(np.mean(train_values)); median = float(np.median(train_values))
    if not np.allclose(aligned.training_mean_prediction.to_numpy(np.float64), mean, rtol=0, atol=1e-15): raise ValueError("Training-mean reconstruction mismatch")
    aligned["training_mean"] = mean; aligned["training_median"] = median; aligned["ssi_raw_euclidean"] = aligned.unscaled_ssi_euclidean_distance; aligned["ssi_isotonic"] = aligned.calibrated_ssi_prediction; aligned["pairwise_random_forest_ssi"] = aligned.SSI_random_forest_prediction; aligned["true_jaccard_distance"] = target_values
    aligned = aligned[ALIGNED_COLUMNS]
    numeric = aligned[["true_jaccard_distance", *ALL_METHODS]].to_numpy(np.float64)
    if not np.isfinite(numeric).all() or (aligned.ssi_raw_euclidean < 0).any(): raise ValueError("Nonfinite or negative aligned prediction")
    for method in PREDICTIVE_METHODS:
        if ((aligned[method] < 0) | (aligned[method] > 1)).any(): raise ValueError(f"Prediction range failure: {method}")
    return training, aligned, mean, median

def metric_rows(aligned: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str, str, str], float]]:
    """Calculate all dataset metrics and return a lookup."""
    rows = []; lookup = {}
    for (holdout, condition, role, dataset), group in aligned.groupby(["outer_holdout", "data_condition", "pair_role", "dataset"], sort=False):
        target = group.true_jaccard_distance.to_numpy(np.float64)
        for method in ALL_METHODS:
            metrics = point_metrics(target, group[method].to_numpy(np.float64), method in PREDICTIVE_METHODS)
            for metric, (estimate, reason) in metrics.items():
                rows.append({"outer_holdout": holdout, "data_condition": condition, "pair_role": role, "dataset": dataset, "method": method, "metric": metric, "estimate": estimate, "n_pairs": len(group), "missing_reason": reason}); lookup[(holdout, condition, role, dataset, method, metric)] = estimate
    return pd.DataFrame(rows, columns=METRIC_COLUMNS), lookup

def macro_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate unweighted familiar ecosystem macros without row pooling."""
    rows = []
    familiar = metrics[metrics.pair_role == "familiar_test"]
    for (holdout, condition, method, metric), group in familiar.groupby(["outer_holdout", "data_condition", "method", "metric"], sort=False):
        defined = group[np.isfinite(group.estimate.to_numpy(np.float64))]
        estimate = float(np.mean(defined.estimate)) if len(defined) == 3 else math.nan
        rows.append({"outer_holdout": holdout, "data_condition": condition, "pair_role": "familiar_test", "dataset": "familiar_macro", "method": method, "metric": metric, "estimate": estimate, "component_count": 3, "defined_component_count": len(defined), "n_pairs": int(group.n_pairs.sum()), "missing_reason": "" if len(defined) == 3 else "component_metric_missing"})
    return pd.DataFrame(rows)

def bootstrap_rows(aligned: pd.DataFrame, lookup: dict[tuple[str, str, str, str, str, str], float]) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str], dict[tuple[str, str], np.ndarray]]]:
    """Calculate all primary event-node intervals and familiar macros."""
    rows = []; cache = {}
    for key, group in aligned.groupby(["outer_holdout", "data_condition", "pair_role", "dataset"], sort=False):
        holdout, condition, role, dataset = key; values = {method: group[method].to_numpy(np.float64) for method in ALL_METHODS}; bootstrap, n_events = bootstrap_group(group, values); cache[key] = bootstrap; seed = group_seed(*key); stream = f"sha256_first16:{seed}"
        for (method, metric), replicates in bootstrap.items():
            lower, upper, valid, undefined, reason = percentile(replicates); point = lookup[(holdout, condition, role, dataset, method, metric)]
            rows.append({"outer_holdout": holdout, "data_condition": condition, "pair_role": role, "dataset": dataset, "method": method, "metric": metric, "estimate": point, "ci_lower": lower, "ci_upper": upper, "n_pairs": len(group), "n_events": n_events, "requested_replicates": 2000, "valid_replicates": valid, "undefined_replicates": undefined, "stream_seed": seed, "stream_id": stream, "missing_reason": reason})
    for holdout in HOLDOUTS:
        for condition in CONDITIONS:
            datasets = [x for x in HOLDOUTS if x != holdout]
            for method in PREDICTIVE_METHODS:
                arrays = [cache[(holdout, condition, "familiar_test", dataset)][(method, "mae")] for dataset in datasets]; macro = np.mean(np.vstack(arrays), axis=0); lower, upper, valid, undefined, reason = percentile(macro); point = float(np.mean([lookup[(holdout, condition, "familiar_test", dataset, method, "mae")] for dataset in datasets]))
                rows.append({"outer_holdout": holdout, "data_condition": condition, "pair_role": "familiar_test", "dataset": "familiar_macro", "method": method, "metric": "mae", "estimate": point, "ci_lower": lower, "ci_upper": upper, "n_pairs": int(sum(len(aligned[(aligned.outer_holdout == holdout) & (aligned.data_condition == condition) & (aligned.pair_role == "familiar_test") & (aligned.dataset == dataset)]) for dataset in datasets)), "n_events": "", "requested_replicates": 2000, "valid_replicates": valid, "undefined_replicates": undefined, "stream_seed": "", "stream_id": "independent_dataset_streams_unweighted_macro", "missing_reason": reason})
            for method in RANKING_METHODS:
                stack = np.vstack([cache[(holdout, condition, "familiar_test", dataset)][(method, "spearman")] for dataset in datasets]); macro = np.where(np.all(np.isfinite(stack), axis=0), np.mean(stack, axis=0), np.nan); lower, upper, valid, undefined, reason = percentile(macro); points = [lookup[(holdout, condition, "familiar_test", dataset, method, "spearman")] for dataset in datasets]; point = float(np.mean(points)) if np.isfinite(points).all() else math.nan
                rows.append({"outer_holdout": holdout, "data_condition": condition, "pair_role": "familiar_test", "dataset": "familiar_macro", "method": method, "metric": "spearman", "estimate": point, "ci_lower": lower, "ci_upper": upper, "n_pairs": int(sum(len(aligned[(aligned.outer_holdout == holdout) & (aligned.data_condition == condition) & (aligned.pair_role == "familiar_test") & (aligned.dataset == dataset)]) for dataset in datasets)), "n_events": "", "requested_replicates": 2000, "valid_replicates": valid, "undefined_replicates": undefined, "stream_seed": "", "stream_id": "independent_dataset_streams_unweighted_macro", "missing_reason": reason})
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS), cache

def seed_metric_rows(aligned: pd.DataFrame) -> pd.DataFrame:
    """Calculate secondary individual-seed point metrics and summaries."""
    sealed = pd.read_csv(ROOT / "results/final_fit_and_seal/siamese_seed_predictions.csv", float_precision="round_trip"); rows = []
    for key, group in aligned.groupby(["outer_holdout", "data_condition", "pair_role", "dataset"], sort=False):
        holdout, condition, role, dataset = key; target = group.set_index("pair_id").true_jaccard_distance
        subset = sealed[(sealed.outer_holdout == holdout) & (sealed.data_condition == condition) & (sealed.pair_role == role) & (sealed.dataset == dataset)]
        seed_values = {}
        for seed in (42, 43, 44, 45, 46):
            part = subset[subset.seed == seed].set_index("pair_id").loc[target.index]
            if len(part) != len(target) or not np.array_equal(part.event_uid_a.to_numpy(), group.set_index("pair_id").loc[target.index].event_uid_a.to_numpy()): raise ValueError("Siamese seed alignment mismatch")
            metrics = point_metrics(target.to_numpy(np.float64), part.prediction.to_numpy(np.float64), True); metrics["prediction_range"] = (float(part.prediction.max() - part.prediction.min()), "")
            for metric in ["mae", "rmse", "r2", "spearman", "kendall_tau_b", "prediction_mean", "prediction_standard_deviation", "prediction_range", "calibration_intercept", "calibration_slope"]:
                estimate, reason = metrics[metric]; rows.append({"outer_holdout": holdout, "data_condition": condition, "pair_role": role, "dataset": dataset, "seed": seed, "summary_statistic": "individual_seed", "metric": metric, "estimate": estimate, "n_pairs": len(group), "missing_reason": reason}); seed_values.setdefault(metric, []).append(estimate)
        ensemble_metrics = point_metrics(target.to_numpy(np.float64), group.siamese_ensemble_ssi.to_numpy(np.float64), True); ensemble_metrics["prediction_range"] = (float(group.siamese_ensemble_ssi.max() - group.siamese_ensemble_ssi.min()), "")
        for metric, values in seed_values.items():
            array = np.asarray(values, np.float64); finite = array[np.isfinite(array)]
            for statistic, estimate in (("ensemble", ensemble_metrics[metric][0]), ("median_seed", float(np.median(finite)) if len(finite) else math.nan), ("minimum_seed", float(np.min(finite)) if len(finite) else math.nan), ("maximum_seed", float(np.max(finite)) if len(finite) else math.nan), ("between_seed_standard_deviation", float(np.std(finite, ddof=0)) if len(finite) else math.nan)):
                rows.append({"outer_holdout": holdout, "data_condition": condition, "pair_role": role, "dataset": dataset, "seed": "", "summary_statistic": statistic, "metric": metric, "estimate": estimate, "n_pairs": len(group), "missing_reason": "" if math.isfinite(estimate) else "all_seed_metrics_undefined"})
    return pd.DataFrame(rows)

def gate_rows(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Apply the three locked filtered unseen-ranking gates."""
    rows = []
    for method in ("gdm", "pairwise_random_forest_ssi", "siamese_ensemble_ssi"):
        points = []; lowers = []
        for holdout in HOLDOUTS:
            point_row = metrics[(metrics.outer_holdout == holdout) & (metrics.data_condition == "filtered") & (metrics.pair_role == "unseen_test") & (metrics.method == method) & (metrics.metric == "spearman")]
            interval = bootstrap[(bootstrap.outer_holdout == holdout) & (bootstrap.data_condition == "filtered") & (bootstrap.pair_role == "unseen_test") & (bootstrap.method == method) & (bootstrap.metric == "spearman")]
            if len(point_row) != 1 or len(interval) != 1: raise ValueError("Gate input coverage mismatch")
            point = float(point_row.iloc[0].estimate); lower = float(interval.iloc[0].ci_lower); points.append(point); lowers.append(lower); rows.append({"method": method, "record_type": "ecosystem_input", "outer_holdout": holdout, "unseen_spearman": point, "ci_lower": lower, "median_unseen_spearman": "", "positive_ci_lower_count": "", "gate_result": "", "notes": "filtered unseen only"})
        median = float(np.median(points)); positive = int(sum(math.isfinite(value) and value > 0 for value in lowers)); passed = median >= 0.20 and positive >= 3
        rows.append({"method": method, "record_type": "gate_summary", "outer_holdout": "four_holdout_summary", "unseen_spearman": "", "ci_lower": "", "median_unseen_spearman": median, "positive_ci_lower_count": positive, "gate_result": "PASS" if passed else "FAIL", "notes": "Requires median>=0.20 and at least 3/4 CI lower limits >0"})
    return pd.DataFrame(rows)

def failure_rows(aligned: pd.DataFrame, metrics: pd.DataFrame, macro: pd.DataFrame, trainings: dict[tuple[str, str], pd.DataFrame], features: pd.DataFrame) -> pd.DataFrame:
    """Create the target-informed post-evaluation failure decomposition."""
    rows = []
    metric_lookup = {(r.outer_holdout, r.data_condition, r.pair_role, r.dataset, r.method, r.metric): r.estimate for r in metrics.itertuples(index=False)}; macro_lookup = {(r.outer_holdout, r.data_condition, r.method, r.metric): r.estimate for r in macro.itertuples(index=False)}
    for (holdout, condition, role, dataset), group in aligned.groupby(["outer_holdout", "data_condition", "pair_role", "dataset"], sort=False):
        target = group.true_jaccard_distance.to_numpy(np.float64); train_target = trainings[(holdout, condition)].jaccard_distance.to_numpy(np.float64); train_ids = sorted(set(trainings[(holdout, condition)].event_uid_a) | set(trainings[(holdout, condition)].event_uid_b)); eval_ids = sorted(set(group.event_uid_a) | set(group.event_uid_b))
        common = {"outer_holdout": holdout, "data_condition": condition, "pair_role": role, "dataset": dataset}
        target_values = {"target_mean_difference_from_final_train": float(np.mean(target) - np.mean(train_target)), "target_median_difference_from_final_train": float(np.median(target) - np.median(train_target)), "target_ks_statistic": float(ks_2samp(train_target, target).statistic), "target_wasserstein_distance": wasserstein_1d(train_target, target)}
        for metric, estimate in target_values.items(): rows.append({**common, "category": "target_distribution_shift", "method": "target", "feature": "", "metric": metric, "estimate": estimate, "interpretation_label": "target_informed_post_evaluation", "notes": "Diagnostic only; not model selection."})
        train_x = features.loc[train_ids, SSI_COLUMNS].to_numpy(np.float64); eval_x = features.loc[eval_ids, SSI_COLUMNS].to_numpy(np.float64)
        for index, feature in enumerate(SSI_COLUMNS):
            train_sd = float(np.std(train_x[:, index], ddof=0)); values = {"ssi_ks_statistic": float(ks_2samp(train_x[:, index], eval_x[:, index]).statistic), "ssi_wasserstein_distance": wasserstein_1d(train_x[:, index], eval_x[:, index]), "ssi_standardized_mean_difference": (float(np.mean(eval_x[:, index]) - np.mean(train_x[:, index])) / train_sd if train_sd > 0 else math.nan)}
            for metric, estimate in values.items(): rows.append({**common, "category": "ssi_source_to_target_shift", "method": "SSI", "feature": feature, "metric": metric, "estimate": estimate, "interpretation_label": "", "notes": "Unique events; raw SSI; final_train source versus evaluated events."})
        for method in ("ssi_isotonic", "gdm", "pairwise_random_forest_ssi", "siamese_ensemble_ssi"):
            prediction = group[method].to_numpy(np.float64); target_sd = float(np.std(target, ddof=0)); prediction_sd = float(np.std(prediction, ddof=0)); target_range = float(np.ptp(target)); ratio = prediction_sd / target_sd if target_sd > 0 else math.nan; range_ratio = float(np.ptp(prediction)) / target_range if target_range > 0 else math.nan; calibration_slope = metric_lookup[(holdout, condition, role, dataset, method, "calibration_slope")]; spearman = metric_lookup[(holdout, condition, role, dataset, method, "spearman")]
            for metric, estimate, label in (("prediction_sd_divided_by_target_sd", ratio, compression_label(ratio)), ("prediction_range_divided_by_target_range", range_ratio, ""), ("calibration_slope", calibration_slope, ""), ("within_dataset_spearman", spearman, "")):
                rows.append({**common, "category": "prediction_behavior", "method": method, "feature": "", "metric": metric, "estimate": estimate, "interpretation_label": label, "notes": "Target-informed post-evaluation failure decomposition."})
    for holdout in HOLDOUTS:
        for condition in CONDITIONS:
            unseen_dataset = holdout
            for method in PREDICTIVE_METHODS:
                for metric in ("mae", "rmse", "r2", "spearman", "calibration_slope", "prediction_sd_divided_by_target_sd"):
                    familiar = macro_lookup.get((holdout, condition, method, metric), math.nan); unseen = metric_lookup.get((holdout, condition, "unseen_test", unseen_dataset, method, metric), math.nan)
                    if metric == "calibration_slope": estimate = abs(unseen - 1) - abs(familiar - 1) if math.isfinite(unseen) and math.isfinite(familiar) else math.nan
                    elif metric == "prediction_sd_divided_by_target_sd": estimate = unseen - familiar if math.isfinite(unseen) and math.isfinite(familiar) else math.nan
                    else: estimate = unseen - familiar if math.isfinite(unseen) and math.isfinite(familiar) else math.nan
                    rows.append({"outer_holdout": holdout, "data_condition": condition, "pair_role": "unseen_minus_familiar_macro", "dataset": unseen_dataset, "category": "familiar_to_unseen_degradation", "method": method, "feature": "", "metric": metric, "estimate": estimate, "interpretation_label": "", "notes": "Direction retained; calibration metric is change in absolute slope deviation from 1."})
    return pd.DataFrame(rows)

def target_stability_rows() -> pd.DataFrame:
    """Calculate the locked deduplicated same-site date-gap target summaries."""
    event = pd.read_csv(ROOT / "data_clean/event_features.csv", usecols=["event_uid", "dataset", "site", "event_datetime"], keep_default_na=False); event = event.set_index("event_uid"); rows = []
    for condition in CONDITIONS:
        frames = [pd.read_csv(ROOT / f"experiments/holdout_{holdout}/{condition}/outer_pairs.csv", usecols=["pair_id", "dataset", "event_uid_a", "event_uid_b", "jaccard_distance"], float_precision="round_trip") for holdout in HOLDOUTS]
        pairs = pd.concat(frames, ignore_index=True).drop_duplicates("pair_id").sort_values("pair_id")
        for dataset in HOLDOUTS:
            subset = pairs[pairs.dataset == dataset].copy()
            if dataset == "usa":
                for gap in ["same_date", "1_to_7_days", "8_to_30_days", "31_to_90_days", "more_than_90_days"]: rows.append({"dataset": dataset, "data_condition": condition, "date_gap_group": gap, "metric": "availability", "estimate": "", "n_pairs": 0, "sufficiently_populated": False, "missing_reason": "site_identifier_unavailable"})
                continue
            a = event.loc[subset.event_uid_a]; b = event.loc[subset.event_uid_b]; same_site = (a.site.to_numpy(str) != "") & (a.site.to_numpy(str) == b.site.to_numpy(str)); dates_a = pd.DatetimeIndex(pd.to_datetime(a.event_datetime.to_numpy(), errors="coerce")); dates_b = pd.DatetimeIndex(pd.to_datetime(b.event_datetime.to_numpy(), errors="coerce")); valid = same_site & dates_a.notna() & dates_b.notna(); subset = subset.loc[valid].copy(); days = np.abs((dates_a[valid] - dates_b[valid]).total_seconds().to_numpy() / 86400.0)
            labels = np.select([days == 0, (days >= 1) & (days <= 7), (days > 7) & (days <= 30), (days > 30) & (days <= 90), days > 90], ["same_date", "1_to_7_days", "8_to_30_days", "31_to_90_days", "more_than_90_days"], default="unclassified"); subset["gap"] = labels
            for gap in ["same_date", "1_to_7_days", "8_to_30_days", "31_to_90_days", "more_than_90_days"]:
                values = subset.loc[subset.gap == gap, "jaccard_distance"].to_numpy(np.float64); sufficient = len(values) >= 30
                stats = {"count": float(len(values)), "mean": float(np.mean(values)) if len(values) else math.nan, "median": float(np.median(values)) if len(values) else math.nan, "standard_deviation": float(np.std(values, ddof=0)) if len(values) else math.nan, "interquartile_range": float(np.quantile(values, .75) - np.quantile(values, .25)) if len(values) else math.nan, "minimum": float(np.min(values)) if len(values) else math.nan, "maximum": float(np.max(values)) if len(values) else math.nan}
                for metric, estimate in stats.items(): rows.append({"dataset": dataset, "data_condition": condition, "date_gap_group": gap, "metric": metric, "estimate": estimate, "n_pairs": len(values), "sufficiently_populated": sufficient, "missing_reason": "" if len(values) else "no_eligible_same_site_pairs"})
    return pd.DataFrame(rows)
