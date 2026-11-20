import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from analysis.common import ROOT, CONFIG, HOLDOUTS, CONDITIONS, SSI, IDENTIFIERS, read, features, cells, output_folder, write_csv
from analysis import fit_models
from analysis.random_forest_utils import symmetric_pair_representation, random_forest


def run_cell(holdout, condition, models, out):
    folder = ROOT / f'experiments/holdout_{holdout}/{condition}'
    pairs = read(folder / 'outer_pairs.csv')
    train = pairs[pairs.pair_role == 'final_train']
    test = pairs[pairs.pair_role != 'final_train'].sort_values(['pair_role', 'dataset', 'pair_id'], kind='mergesort').reset_index(drop=True)
    frame = test[IDENTIFIERS].copy()
    frame['true_jaccard_distance'] = test.jaccard_distance
    destination = out / f'predictions/{holdout}__{condition}.csv'
    if destination.exists():
        previous = read(destination)
        if previous.pair_id.tolist() != frame.pair_id.tolist():
            raise ValueError('Existing reproduction predictions use different pairs')
        frame = previous
    exact = features(True)
    targets = train.jaccard_distance.to_numpy(float)
    a = exact.loc[train.event_uid_a, SSI].to_numpy(float)
    b = exact.loc[train.event_uid_b, SSI].to_numpy(float)
    ta = exact.loc[test.event_uid_a, SSI].to_numpy(float)
    tb = exact.loc[test.event_uid_b, SSI].to_numpy(float)
    scaler = StandardScaler().fit(np.vstack((a, b)))
    train_distance = np.linalg.norm(scaler.transform(a) - scaler.transform(b), axis=1)
    test_distance = np.linalg.norm(scaler.transform(ta) - scaler.transform(tb), axis=1)
    if 'baselines' in models:
        isotonic = IsotonicRegression(increasing=True, out_of_bounds='clip', y_min=0, y_max=1)
        isotonic.fit(train_distance, targets)
        frame['training_mean'] = targets.mean()
        frame['training_median'] = np.median(targets)
        frame['ssi_raw_euclidean'] = np.linalg.norm(ta - tb, axis=1)
        frame['ssi_isotonic'] = isotonic.predict(test_distance)
    if 'forest' in models:
        selected = read(ROOT / f'results/pairwise_random_forest/holdout_{holdout}/{condition}/selected_hyperparameters.csv')
        row = selected.loc[selected.feature_regime == 'SSI'].iloc[0]
        candidate = {'max_depth': None if row.max_depth == 'none' else int(row.max_depth),
                     'min_samples_leaf': int(row.min_samples_leaf), 'max_features': str(row.max_features) if str(row.max_features) == 'sqrt' else float(row.max_features)}
        model = random_forest(CONFIG['forest']['fixed_parameters'], candidate)
        model.fit(symmetric_pair_representation(scaler.transform(a), scaler.transform(b)), targets)
        frame['pairwise_random_forest_ssi'] = model.predict(symmetric_pair_representation(scaler.transform(ta), scaler.transform(tb)))
    cell_dict = next(x for x in CONFIG['final_cells'] if x['holdout'] == holdout and x['condition'] == condition)
    cell = (holdout, condition, cell_dict['preprocessing'], cell_dict['architecture'], cell_dict['epochs'])
    manifest, final_targets = fit_models.load_manifest(folder / 'outer_pairs.csv')
    original_precision = features(False)
    for family in ['gdm', 'siamese']:
        if family not in models:
            continue
        if family == 'gdm':
            prediction, parameters, ledger = fit_models.run_gdm_fit(cell, manifest, final_targets, original_precision, out / f'checkpoints/gdm/{holdout}__{condition}.rds')
            column = prediction.set_index('pair_id').prediction
            frame['gdm'] = frame.pair_id.map(column)
            write_csv(parameters, out / f'parameters/gdm__{holdout}__{condition}.csv')
            (out / f'parameters/gdm__{holdout}__{condition}__ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
        else:
            seeds = []
            for seed in CONFIG['seeds']:
                prediction, parameters, history, ledger = fit_models.run_siamese_fit(cell, seed, manifest, final_targets, original_precision, out / f'checkpoints/siamese/{holdout}__{condition}__{seed}.pt')
                seeds.append(prediction.set_index('pair_id').prediction.reindex(frame.pair_id).to_numpy(float))
                write_csv(prediction, out / f'seed_predictions/{holdout}__{condition}__{seed}.csv')
                write_csv(parameters, out / f'parameters/siamese__{holdout}__{condition}__{seed}.csv')
                write_csv(pd.DataFrame(history), out / f'parameters/siamese__{holdout}__{condition}__{seed}__history.csv')
                (out / f'parameters/siamese__{holdout}__{condition}__{seed}__ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
            frame['siamese_ensemble_ssi'] = np.mean(np.vstack(seeds), axis=0)
    if frame.isna().any().any():
        raise ValueError('Missing aligned predictions')
    write_csv(frame, destination)
    frozen = read(ROOT / 'results/final_evaluation/aligned_predictions.csv')
    reference = frozen[(frozen.outer_holdout == holdout) & (frozen.data_condition == condition)].set_index('pair_id').reindex(frame.pair_id)
    differences = {name: float(np.max(np.abs(frame[name].to_numpy(float) - reference[name].to_numpy(float))))
                   for name in frame if name not in IDENTIFIERS and name != 'true_jaccard_distance'}
    (out / f'comparison__{holdout}__{condition}.json').write_text(json.dumps(differences, indent=2) + '\n')
    print(holdout, condition, differences, flush=True)
    if max(differences.values(), default=0.0) > 1e-12:
        print('Numerical refit differences detected. See RELEASE_CHECKS.md; archived results were preserved.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Refit the published SSI models using their selected settings.')
    parser.add_argument('--models', nargs='+', choices=['all', 'baselines', 'forest', 'gdm', 'siamese'], default=['all'])
    parser.add_argument('--holdout', choices=['all', *HOLDOUTS], default='all')
    parser.add_argument('--condition', choices=['both', *CONDITIONS], default='both')
    parser.add_argument('--output', default='reproduced')
    args = parser.parse_args()
    models = ['baselines', 'forest', 'gdm', 'siamese'] if 'all' in args.models else args.models
    out = output_folder(args.output)
    started = time.time()
    for h, c in cells(args.holdout, args.condition):
        run_cell(h, c, models, out)
    print(f'Completed in {time.time() - started:.1f} seconds. Original results were preserved.')


if __name__ == '__main__':
    main()
