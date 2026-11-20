import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from analysis.common import ROOT, SSI, read, features, cells, output_folder, write_csv
from analysis import evaluation
from analysis.final_evaluation_utils import PREDICTIVE_METHODS, fixture_check


def audit(out):
    parts = []
    training = {}
    for h, c in cells():
        train, frame, _, _ = evaluation.align_condition(h, c, evaluation.community_map(c))
        training[(h, c)] = train
        parts.append(frame)
    aligned = pd.concat(parts, ignore_index=True)
    frozen = read(ROOT / 'results/final_evaluation/aligned_predictions.csv')
    keys = ['outer_holdout', 'data_condition', 'pair_role', 'dataset', 'pair_id']
    pd.testing.assert_frame_equal(aligned.sort_values(keys).reset_index(drop=True), frozen.sort_values(keys).reset_index(drop=True), check_dtype=False, check_exact=True)
    metrics, lookup = evaluation.metric_rows(aligned)
    saved = read(ROOT / 'results/final_evaluation/metrics_by_dataset.csv')
    metric_keys = ['outer_holdout', 'data_condition', 'pair_role', 'dataset', 'method', 'metric']
    a = metrics.set_index(metric_keys).sort_index()
    b = saved.set_index(metric_keys).sort_index()
    av = pd.to_numeric(a.estimate, errors='coerce').to_numpy(float)
    bv = pd.to_numeric(b.estimate, errors='coerce').to_numpy(float)
    np.testing.assert_allclose(av, bv, rtol=5e-15, atol=1e-12, equal_nan=True)
    write_csv(aligned, out / 'aligned_predictions.csv')
    write_csv(metrics, out / 'metrics_by_dataset.csv')
    write_csv(evaluation.macro_rows(metrics), out / 'familiar_macro_metrics.csv')
    signed = []
    ks_count = 0
    maximum_ks_difference = 0.0
    f = features(False)
    diagnostic = read(ROOT / 'results/final_evaluation/failure_decomposition.csv')
    group_keys = ['outer_holdout', 'data_condition', 'pair_role', 'dataset']
    for key, frame in aligned.groupby(group_keys):
        h, c, role, dataset = key
        target = frame.true_jaccard_distance.to_numpy(float)
        for method in PREDICTIVE_METHODS:
            prediction = frame[method].to_numpy(float)
            signed.append(dict(zip(group_keys, key)) | {'method': method, 'n_pairs': len(frame), 'mean_signed_error': float(np.mean(prediction - target))})
        train = training[(h, c)]
        train_ids = sorted(set(train.event_uid_a) | set(train.event_uid_b))
        test_ids = sorted(set(frame.event_uid_a) | set(frame.event_uid_b))
        saved_ks = diagnostic[(diagnostic.outer_holdout == h) & (diagnostic.data_condition == c) & (diagnostic.pair_role == role) & (diagnostic.dataset == dataset)]
        checks = [('target_ks_statistic', '', train.jaccard_distance.to_numpy(float), target)]
        checks += [('ssi_ks_statistic', column, f.loc[train_ids, column].to_numpy(float), f.loc[test_ids, column].to_numpy(float)) for column in SSI]
        for metric, column, x, y in checks:
            values = np.unique(np.concatenate((x, y)))
            distance = float(np.max(np.abs(np.searchsorted(np.sort(x), values, side='right') / len(x) - np.searchsorted(np.sort(y), values, side='right') / len(y))))
            scipy_distance = float(ks_2samp(x, y).statistic)
            expected = float(saved_ks.loc[(saved_ks.metric == metric) & (saved_ks.feature == column), 'estimate'].iloc[0])
            difference = max(abs(distance - expected), abs(distance - scipy_distance))
            if difference > 1e-12:
                raise ValueError('KS statistic mismatch')
            maximum_ks_difference = max(maximum_ks_difference, difference)
            ks_count += 1
    signed = pd.DataFrame(signed)
    write_csv(signed, out / 'mean_signed_error.csv')
    macro = signed[signed.pair_role == 'familiar_test'].groupby(['outer_holdout', 'data_condition', 'method']).mean_signed_error.mean().reset_index()
    write_csv(macro, out / 'familiar_macro_signed_error.csv')
    reference = read(ROOT / 'publication/tables/revision_mean_signed_error.csv').set_index(group_keys + ['method']).sort_index()
    actual = signed.set_index(group_keys + ['method']).sort_index()
    np.testing.assert_allclose(actual.mean_signed_error, reference.mean_signed_error, rtol=0, atol=1e-14)
    fixture_check()
    report = {'prediction_rows': len(aligned), 'metric_rows': len(metrics), 'maximum_metric_difference': float(np.nanmax(np.abs(av - bv))), 'ks_checks': ks_count, 'maximum_ks_difference': maximum_ks_difference, 'signed_error_rows': len(signed), 'status': 'PASS'}
    (out / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return aligned, metrics, lookup


def main():
    parser = argparse.ArgumentParser(description='Recalculate and verify published point metrics, signed errors and KS diagnostics.')
    parser.add_argument('--bootstrap', action='store_true', help='Also repeat the original 2000 event bootstrap draws per group.')
    parser.add_argument('--output', default='reproduced/evaluation')
    args = parser.parse_args()
    out = output_folder(args.output)
    aligned, metrics, lookup = audit(out)
    if args.bootstrap:
        intervals, _ = evaluation.bootstrap_rows(aligned, lookup)
        write_csv(intervals, out / 'bootstrap_intervals.csv')
        write_csv(evaluation.gate_rows(metrics, intervals), out / 'unseen_ranking_gate_results.csv')
        print('Completed the original event bootstrap and transfer criterion.')


if __name__ == '__main__':
    main()
