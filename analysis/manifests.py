from __future__ import annotations

import hashlib

import io

import math

import random

from bisect import bisect_right

from pathlib import Path

from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

PROTECTED_DIRS = (
    PROJECT_ROOT / "original_downloads",
    PROJECT_ROOT / "sarabsethi-sscape-avian-div-generalisability-d9f2547",
    PROJECT_ROOT / "data_clean",
)

BASE_COLUMNS = ["event_uid", "dataset", "base_role", "random_seed"]

PAIR_COLUMNS = [
    "pair_id",
    "outer_holdout",
    "data_condition",
    "inner_validation_dataset",
    "pair_role",
    "dataset",
    "event_uid_a",
    "event_uid_b",
    "jaccard_distance",
    "random_seed",
]

COUNT_COLUMNS = [
    "manifest",
    "inner_validation_dataset",
    "pair_role",
    "dataset",
    "eligible_event_count",
    "available_pair_count",
    "selected_pair_count",
    "balance_target",
    "balancing_scope",
]

def derived_seed(root_seed: int, context: str) -> int:
    """Derive an order-independent deterministic random substream seed."""
    digest = hashlib.sha256(f"{root_seed}|{context}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")

def create_base_split(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Assign one permanent development or familiar-test role per event."""
    rows: list[dict[str, Any]] = []
    root_seed = int(config["random_seed"])
    fraction = float(config["base_split"]["development_fraction"])
    for dataset in config["datasets"]:
        event_uids = sorted(features.loc[features["dataset"] == dataset, "event_uid"].astype(str))
        shuffled = list(event_uids)
        random.Random(derived_seed(root_seed, f"base_split|{dataset}")).shuffle(shuffled)
        development_count = math.floor(len(shuffled) * fraction)
        development_ids = set(shuffled[:development_count])
        for uid in event_uids:
            rows.append(
                {
                    "event_uid": uid,
                    "dataset": dataset,
                    "base_role": "development" if uid in development_ids else "familiar_test",
                    "random_seed": root_seed,
                }
            )
    result = pd.DataFrame(rows, columns=BASE_COLUMNS)
    if len(result) != len(features) or not result["event_uid"].is_unique:
        raise ValueError("Permanent base split does not contain every event exactly once")
    return result

def dataframe_text(frame: pd.DataFrame) -> str:
    """Serialize a manifest deterministically."""
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, float_format="%.17g", lineterminator="\n")
    return buffer.getvalue()

def condition_species_sets(
    original: dict[str, frozenset[str]], condition: str
) -> tuple[dict[str, frozenset[str]], dict[str, bool], dict[str, str], dict[str, bool]]:
    """Apply only the configured condition eligibility and exact-NA rule."""
    labels_by_uid: dict[str, frozenset[str]] = {}
    eligibility: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    na_removed: dict[str, bool] = {}
    for uid, labels in original.items():
        if condition == "author_compatible":
            labels_by_uid[uid] = labels
            eligibility[uid] = True
            reasons[uid] = ""
            na_removed[uid] = False
        elif condition == "filtered":
            eligible = bool(labels)
            eligibility[uid] = eligible
            reasons[uid] = "" if eligible else "empty_community"
            removed = "NA" in labels
            na_removed[uid] = removed
            labels_by_uid[uid] = frozenset(label for label in labels if label != "NA")
            if eligible and not labels_by_uid[uid]:
                raise ValueError(f"Filtered event unexpectedly became empty after exact NA removal: {uid}")
        else:
            raise ValueError(f"Unsupported data condition: {condition}")
    return labels_by_uid, eligibility, reasons, na_removed

def create_event_roles(
    features: pd.DataFrame,
    base_split: pd.DataFrame,
    original_species: dict[str, frozenset[str]],
    condition_species: dict[str, frozenset[str]],
    eligibility: dict[str, bool],
    exclusion_reasons: dict[str, str],
    na_removed: dict[str, bool],
    holdout: str,
    condition: str,
    datasets: list[str],
    seed: int,
) -> pd.DataFrame:
    """Create one eligibility and outer-role row per central event."""
    base_by_uid = base_split.set_index("event_uid")["base_role"].to_dict()
    source_datasets = [dataset for dataset in datasets if dataset != holdout]
    rows: list[dict[str, Any]] = []
    for row in features.itertuples(index=False):
        uid = str(row.event_uid)
        dataset = str(row.dataset)
        base_role = str(base_by_uid[uid])
        outer_role = "unseen_test" if dataset == holdout else base_role
        record: dict[str, Any] = {
            "event_uid": uid,
            "dataset": dataset,
            "source_event_id": str(row.source_event_id),
            "outer_holdout": holdout,
            "data_condition": condition,
            "base_role": base_role,
            "outer_role": outer_role,
            "eligible": eligibility[uid],
            "exclusion_reason": exclusion_reasons[uid],
            "original_species_count": len(original_species[uid]),
            "condition_species_count": len(condition_species[uid]),
            "exact_na_removed": na_removed[uid],
            "random_seed": seed,
        }
        for validation_dataset in datasets:
            column = f"inner_role_when_validation_{validation_dataset}"
            if validation_dataset not in source_datasets:
                record[column] = "not_applicable"
            elif not eligibility[uid]:
                record[column] = "excluded"
            elif dataset == holdout or base_role != "development":
                record[column] = "not_used"
            elif dataset == validation_dataset:
                record[column] = "validation"
            else:
                record[column] = "train"
        rows.append(record)
    columns = [
        "event_uid",
        "dataset",
        "source_event_id",
        "outer_holdout",
        "data_condition",
        "base_role",
        "outer_role",
        "eligible",
        "exclusion_reason",
        "original_species_count",
        "condition_species_count",
        "exact_na_removed",
        "random_seed",
        *(f"inner_role_when_validation_{dataset}" for dataset in datasets),
    ]
    return pd.DataFrame(rows, columns=columns)

def pair_count(event_count: int) -> int:
    """Return the number of unique unordered pairs."""
    return event_count * (event_count - 1) // 2

def sample_pair_ranks(total: int, selected: int, seed: int) -> list[int]:
    """Sample unique integer pair ranks without constructing candidate pairs."""
    if selected < 0 or selected > total:
        raise ValueError(f"Cannot sample {selected} unique pairs from {total}")
    if selected == total:
        return list(range(total))
    rng = random.Random(seed)
    chosen: set[int] = set()
    for upper in range(total - selected, total):
        candidate = rng.randrange(upper + 1)
        chosen.add(upper if candidate in chosen else candidate)
    if len(chosen) != selected:
        raise RuntimeError("Unique pair-rank sampler produced an incorrect sample size")
    return sorted(chosen)

def unrank_pair(rank: int, event_count: int, starts: list[int]) -> tuple[int, int]:
    """Map a lexicographic unordered-pair rank to two event indexes."""
    first = bisect_right(starts, rank) - 1
    second = first + 1 + rank - starts[first]
    if not 0 <= first < second < event_count:
        raise RuntimeError(f"Invalid unranked pair ({first}, {second}) for rank {rank} and n={event_count}")
    return first, second

def jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    """Calculate exact-set Jaccard distance with defined empty-set behavior."""
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)

def deterministic_pair_id(
    holdout: str,
    condition: str,
    inner_validation: str,
    role: str,
    dataset: str,
    uid_a: str,
    uid_b: str,
) -> str:
    """Create a stable pair identifier from its complete experimental context."""
    value = "|".join((holdout, condition, inner_validation, role, dataset, uid_a, uid_b))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

def select_pair_rows(
    event_uids: Iterable[str],
    selected_count: int,
    root_seed: int,
    context: str,
    holdout: str,
    condition: str,
    inner_validation: str,
    role: str,
    dataset: str,
    species_sets: dict[str, frozenset[str]],
) -> list[dict[str, Any]]:
    """Select and score deterministic unique unordered event pairs."""
    ids = sorted(str(uid) for uid in event_uids)
    total = pair_count(len(ids))
    ranks = sample_pair_ranks(total, selected_count, derived_seed(root_seed, context))
    starts = [index * (2 * len(ids) - index - 1) // 2 for index in range(len(ids))]
    rows: list[dict[str, Any]] = []
    for rank in ranks:
        first, second = unrank_pair(rank, len(ids), starts)
        uid_a = ids[first]
        uid_b = ids[second]
        distance = jaccard_distance(species_sets[uid_a], species_sets[uid_b])
        if not 0.0 <= distance <= 1.0:
            raise ValueError(f"Jaccard distance outside [0, 1] for {uid_a}, {uid_b}: {distance}")
        rows.append(
            {
                "pair_id": deterministic_pair_id(
                    holdout, condition, inner_validation, role, dataset, uid_a, uid_b
                ),
                "outer_holdout": holdout,
                "data_condition": condition,
                "inner_validation_dataset": inner_validation,
                "pair_role": role,
                "dataset": dataset,
                "event_uid_a": uid_a,
                "event_uid_b": uid_b,
                "jaccard_distance": distance,
                "random_seed": root_seed,
            }
        )
    return rows

def eligible_pool(
    event_roles: pd.DataFrame,
    dataset: str,
    base_role: str | None = None,
) -> list[str]:
    """Return sorted eligible IDs for a dataset and optional permanent role."""
    mask = (event_roles["dataset"] == dataset) & event_roles["eligible"]
    if base_role is not None:
        mask &= event_roles["base_role"] == base_role
    return sorted(event_roles.loc[mask, "event_uid"].astype(str))

def append_count_row(
    rows: list[dict[str, Any]],
    manifest: str,
    inner_validation: str,
    role: str,
    dataset: str,
    event_count: int,
    selected: int,
    target: int,
    scope: str,
) -> None:
    """Record available and selected pair counts."""
    rows.append(
        {
            "manifest": manifest,
            "inner_validation_dataset": inner_validation,
            "pair_role": role,
            "dataset": dataset,
            "eligible_event_count": event_count,
            "available_pair_count": pair_count(event_count),
            "selected_pair_count": selected,
            "balance_target": target,
            "balancing_scope": scope,
        }
    )

def build_pair_manifests(
    event_roles: pd.DataFrame,
    species_sets: dict[str, frozenset[str]],
    holdout: str,
    condition: str,
    datasets: list[str],
    root_seed: int,
    unseen_target: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build balanced outer and rotating-inner pair manifests."""
    source_datasets = [dataset for dataset in datasets if dataset != holdout]
    development_pools = {dataset: eligible_pool(event_roles, dataset, "development") for dataset in source_datasets}
    familiar_pools = {dataset: eligible_pool(event_roles, dataset, "familiar_test") for dataset in source_datasets}
    unseen_pool = eligible_pool(event_roles, holdout)
    final_target = min(pair_count(len(values)) for values in development_pools.values())
    familiar_target = min(pair_count(len(values)) for values in familiar_pools.values())
    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    final_train_rows: dict[str, list[dict[str, Any]]] = {}
    for dataset in source_datasets:
        context = f"outer|{holdout}|{condition}|final_train|{dataset}"
        selected_rows = select_pair_rows(
            development_pools[dataset],
            final_target,
            root_seed,
            context,
            holdout,
            condition,
            "",
            "final_train",
            dataset,
            species_sets,
        )
        final_train_rows[dataset] = selected_rows
        outer_rows.extend(selected_rows)
        append_count_row(
            count_rows,
            "outer_pairs",
            "",
            "final_train",
            dataset,
            len(development_pools[dataset]),
            final_target,
            final_target,
            "source_development_minimum",
        )
    for dataset in source_datasets:
        context = f"outer|{holdout}|{condition}|familiar_test|{dataset}"
        outer_rows.extend(
            select_pair_rows(
                familiar_pools[dataset],
                familiar_target,
                root_seed,
                context,
                holdout,
                condition,
                "",
                "familiar_test",
                dataset,
                species_sets,
            )
        )
        append_count_row(
            count_rows,
            "outer_pairs",
            "",
            "familiar_test",
            dataset,
            len(familiar_pools[dataset]),
            familiar_target,
            familiar_target,
            "source_familiar_test_minimum",
        )
    outer_rows.extend(
        select_pair_rows(
            unseen_pool,
            unseen_target,
            root_seed,
            f"outer|{holdout}|{condition}|unseen_test|{holdout}",
            holdout,
            condition,
            "",
            "unseen_test",
            holdout,
            species_sets,
        )
    )
    append_count_row(
        count_rows,
        "outer_pairs",
        "",
        "unseen_test",
        holdout,
        len(unseen_pool),
        unseen_target,
        unseen_target,
        "condition_global_unseen_minimum",
    )
    for validation_dataset in source_datasets:
        for dataset in source_datasets:
            role = "validation" if dataset == validation_dataset else "train"
            for outer_row in final_train_rows[dataset]:
                inner_row = dict(outer_row)
                inner_row["inner_validation_dataset"] = validation_dataset
                inner_row["pair_role"] = role
                inner_rows.append(inner_row)
            append_count_row(
                count_rows,
                "inner_pairs",
                validation_dataset,
                role,
                dataset,
                len(development_pools[dataset]),
                final_target,
                final_target,
                "reuse_outer_final_train_selection",
            )
    return (
        pd.DataFrame(outer_rows, columns=PAIR_COLUMNS),
        pd.DataFrame(inner_rows, columns=PAIR_COLUMNS),
        pd.DataFrame(count_rows, columns=COUNT_COLUMNS),
    )
