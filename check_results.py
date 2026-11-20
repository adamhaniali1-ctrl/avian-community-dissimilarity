import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from analysis.common import ROOT, cells, features, read, output_folder, checksum
from analysis import fit_models
from analysis.final_fit_utils import ARCHITECTURES, SSI_COLUMNS, RawLogitSiamese, apply_preprocessing, predict_model, state_hash, set_deterministic
from prepare_data import verify_files
from evaluate import audit


def checkpoints(out):
    feature_index = features(False)
    seeds = read(ROOT / 'results/final_fit_and_seal/siamese_seed_predictions.csv')
    gdms = read(ROOT / 'results/final_fit_and_seal/gdm_predictions.csv')
    maximum = 0.0
    count = 0
    for h, c in cells():
        manifest, _ = fit_models.load_manifest(ROOT / f'experiments/holdout_{h}/{c}/outer_pairs.csv')
        pairs = manifest[manifest.pair_role != 'final_train'].reset_index(drop=True)
        a, b = fit_models.endpoint_arrays(pairs, feature_index)
        for seed in range(42, 47):
            path = ROOT / f'results/final_fit_and_seal/checkpoints/siamese/holdout_{h}__{c}__seed_{seed}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            set_deterministic(seed)
            model = RawLogitSiamese(ARCHITECTURES[payload['architecture']])
            model.load_state_dict(payload['state_dict'])
            if state_hash(model) != payload['model_state_sha256']:
                raise ValueError('Checkpoint state hash mismatch')
            fit = {key: np.asarray(value, dtype=float) for key, value in payload['preprocessing_parameters'].items()}
            x = apply_preprocessing(payload['preprocessing'], a, fit)
            y = apply_preprocessing(payload['preprocessing'], b, fit)
            _, prediction, swap = predict_model(model, x, y)
            saved = seeds[(seeds.outer_holdout == h) & (seeds.data_condition == c) & (seeds.seed == seed)].set_index('pair_id').loc[pairs.pair_id]
            difference = float(np.max(np.abs(prediction.astype(float) - saved.prediction.to_numpy(float))))
            if difference > 1e-15 or swap != 0:
                raise ValueError(f'Saved model inference differs: {h}, {c}, {seed}: {difference}')
            maximum = max(maximum, difference)
            count += 1
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fit_models.gdm_table(pairs, feature_index, None).to_csv(directory / 'evaluation.csv', index=False, float_format='%.17g')
            saved = gdms[(gdms.outer_holdout == h) & (gdms.data_condition == c)].set_index('pair_id').loc[pairs.pair_id]
            saved[['prediction']].to_csv(directory / 'expected.csv', index=False, float_format='%.17g')
            subprocess.run(['Rscript', str(ROOT / 'analysis/check_gdm.R'), str(ROOT / f'results/final_fit_and_seal/checkpoints/gdm/holdout_{h}__{c}.rds'), str(directory / 'evaluation.csv'), str(directory / 'expected.csv')], check=True, capture_output=True, text=True)
        print(h, c, 'saved checkpoints PASS', flush=True)
    report = {'siamese_checkpoints': count, 'gdm_checkpoints': 8, 'maximum_siamese_prediction_difference': maximum, 'status': 'PASS'}
    (out / 'checkpoint_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Check archived data, results and saved models without refitting.')
    parser.add_argument('--checkpoints', action='store_true')
    parser.add_argument('--output', default='reproduced/checks')
    args = parser.parse_args()
    out = output_folder(args.output)
    print('Verified archived files:', verify_files(), flush=True)
    audit(out)
    if args.checkpoints:
        checkpoints(out)


if __name__ == '__main__':
    main()
