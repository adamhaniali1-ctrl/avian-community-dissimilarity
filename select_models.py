import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from analysis.common import ROOT, CONFIG, HOLDOUTS, CONDITIONS, SSI, cells, features, read, output_folder, write_csv
from analysis.random_forest_utils import candidate_grid, random_forest, symmetric_pair_representation, rank_candidate_rows
from analysis.folds import prepare_fold
from analysis import ssi_preprocessing_comparison_utils as pre
from analysis import ssi_siamese_architecture_utils as arch


def forest(h, c, f, pairs, out):
    candidates = candidate_grid(CONFIG['forest']['parameter_grid'])
    scores = {x['candidate_id']: [] for x in candidates}
    rows = []
    for validation in [x for x in HOLDOUTS if x != h]:
        fold = pairs[pairs.inner_validation_dataset == validation]
        train = fold[fold.pair_role == 'train']
        test = fold[fold.pair_role == 'validation']
        a = f.loc[train.event_uid_a, SSI].to_numpy(float)
        b = f.loc[train.event_uid_b, SSI].to_numpy(float)
        scaler = StandardScaler().fit(np.vstack((a, b)))
        x = symmetric_pair_representation(scaler.transform(a), scaler.transform(b))
        z = symmetric_pair_representation(scaler.transform(f.loc[test.event_uid_a, SSI].to_numpy(float)), scaler.transform(f.loc[test.event_uid_b, SSI].to_numpy(float)))
        for candidate in candidates:
            model = random_forest(CONFIG['forest']['fixed_parameters'], candidate)
            prediction = model.fit(x, train.jaccard_distance).predict(z)
            mae = float(np.mean(np.abs(prediction - test.jaccard_distance.to_numpy(float))))
            scores[candidate['candidate_id']].append(mae)
            rows.append({'candidate_id': candidate['candidate_id'], 'inner_validation_dataset': validation, 'validation_mae': mae})
    summary = [dict(x, mean_validation_mae=float(np.mean(scores[x['candidate_id']])), std_validation_mae=float(np.std(scores[x['candidate_id']], ddof=0))) for x in candidates]
    summary = rank_candidate_rows(summary, 1e-12)
    write_csv(pd.DataFrame(rows), out / f'forest/{h}__{c}__inner_metrics.csv')
    write_csv(pd.DataFrame(summary), out / f'forest/{h}__{c}__selection.csv')
    selected = next(x['candidate_id'] for x in summary if x['selected'])
    frozen = read(ROOT / f'results/pairwise_random_forest/holdout_{h}/{c}/selected_hyperparameters.csv')
    expected = frozen.loc[frozen.feature_regime == 'SSI', 'candidate_id'].iloc[0]
    if selected != expected:
        raise ValueError(f'Forest selection differs: {selected} versus {expected}')
    print(h, c, 'forest', selected, 'PASS', flush=True)


def neural(h, c, family, f, pairs, out, smoke):
    settings = CONFIG['neural_selection']
    locked = next(x for x in CONFIG['final_cells'] if x['holdout'] == h and x['condition'] == c)['preprocessing']
    methods = ['STANDARD', 'ASINH_STANDARD'] if family == 'preprocessing' else [locked]
    records = []
    for method in methods:
        for validation in [x for x in HOLDOUTS if x != h]:
            _, validating, _, arrays, _ = prepare_fold(pairs, validation, f, method)
            for candidate in (['A'] if family == 'preprocessing' else ['B', 'C', 'D']):
                for seed in [42, 43, 44]:
                    fit_id = pre.fit_identifier(h, c, validation, method, seed) if family == 'preprocessing' else arch.fit_identifier(h, c, validation, candidate, method, seed)
                    result = pre.run_fit(fit_id, seed, *arrays, settings) if family == 'preprocessing' else arch.run_fit(fit_id, arch.CANDIDATES[candidate], seed, *arrays, settings)
                    restored = result['restored']
                    record = {'fit_id': fit_id, 'outer_holdout': h, 'data_condition': c, 'inner_validation_dataset': validation, 'candidate': candidate, 'preprocessing_method': method, 'seed': seed, 'validation_mae': restored['mae'], 'best_epoch': result['best_epoch'], 'state_hash': result['state_hash'], 'all_finite': restored['all_finite'], 'degenerate': result['degenerate']}
                    records.append(record)
                    prediction = validating[['pair_id', 'event_uid_a', 'event_uid_b']].copy()
                    prediction['prediction'] = restored['predictions']
                    write_csv(prediction, out / f'{family}/predictions/{fit_id}.csv')
                    write_csv(pd.DataFrame(result['history']), out / f'{family}/history/{fit_id}.csv')
                    write_csv(pd.DataFrame(records), out / f'{family}/{h}__{c}__inner_metrics.csv')
                    reference_path = 'results/ssi_preprocessing_comparison/inner_metrics.csv' if family == 'preprocessing' else 'results/ssi_siamese_architecture_selection/new_inner_metrics.csv'
                    reference = read(ROOT / reference_path)
                    expected = reference.loc[reference.fit_id == fit_id].iloc[0]
                    print(fit_id, 'MAE difference', restored['mae'] - float(expected.validation_mae), flush=True)
                    if smoke:
                        return
    table = pd.DataFrame(records)
    if family == 'architecture':
        source = out / f'preprocessing/{h}__{c}__inner_metrics.csv'
        a = read(source) if source.exists() else read(ROOT / 'results/ssi_preprocessing_comparison/inner_metrics.csv')
        a = a[(a.outer_holdout == h) & (a.data_condition == c) & (a.preprocessing_method == locked)].copy()
        a['candidate'] = 'A'
        if len(a) != 9:
            raise ValueError('Architecture A requires all nine selected preprocessing fits')
        table = pd.concat([a, table], ignore_index=True)
    key = 'preprocessing_method' if family == 'preprocessing' else 'candidate'
    summaries = []
    for value, g in table.groupby(key, sort=False):
        if len(g) != 9:
            raise ValueError('Selection needs nine fits per candidate')
        summaries.append({key: value, 'eligible': bool(g.all_finite.all() and not g.degenerate.any()), 'mean_validation_mae': float(g.validation_mae.mean()), 'sd_validation_mae': float(g.validation_mae.std(ddof=0))})
    eligible = [x for x in summaries if x['eligible']]
    if not eligible:
        raise ValueError('No eligible candidate')
    best = min(x['mean_validation_mae'] for x in eligible)
    tied = [x for x in eligible if abs(x['mean_validation_mae'] - best) <= 1e-12]
    if family == 'architecture':
        sd = min(x['sd_validation_mae'] for x in tied)
        tied = [x for x in tied if abs(x['sd_validation_mae'] - sd) <= 1e-12]
    order = ['STANDARD', 'ASINH_STANDARD'] if family == 'preprocessing' else ['A', 'B', 'C', 'D']
    selected = min(tied, key=lambda x: order.index(x[key]))[key]
    for row in summaries:
        row['selected'] = row[key] == selected
    write_csv(pd.DataFrame(summaries), out / f'{family}/{h}__{c}__selection.csv')
    expected = locked if family == 'preprocessing' else next(x for x in CONFIG['final_cells'] if x['holdout'] == h and x['condition'] == c)['architecture']
    if selected != expected:
        raise ValueError(f'Selection differs: {selected} versus {expected}')
    print(h, c, family, selected, 'PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Repeat SSI model selection using development pairs only.')
    parser.add_argument('--family', required=True, choices=['forest', 'preprocessing', 'architecture'])
    parser.add_argument('--holdout', choices=['all', *HOLDOUTS], default='all')
    parser.add_argument('--condition', choices=['both', *CONDITIONS], default='both')
    parser.add_argument('--smoke', action='store_true', help='Run one neural fit only; this does not perform model selection.')
    parser.add_argument('--output', default='reproduced/selection')
    args = parser.parse_args()
    out = output_folder(args.output)
    f = features() if args.family == 'forest' else pd.read_csv(ROOT / 'data_clean/event_features.csv', usecols=['event_uid', 'dataset', *SSI], keep_default_na=False, low_memory=False, float_precision='round_trip').set_index('event_uid', drop=False)
    for h, c in cells(args.holdout, args.condition):
        pairs = read(ROOT / f'experiments/holdout_{h}/{c}/inner_pairs.csv')
        if args.family == 'forest':
            forest(h, c, f, pairs, out)
        else:
            neural(h, c, args.family, f, pairs, out, args.smoke)
        if args.smoke:
            break


if __name__ == '__main__':
    main()
