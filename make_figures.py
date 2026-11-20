import argparse
import shutil
from analysis.common import ROOT, output_folder
from analysis import plotting as p


def main():
    parser = argparse.ArgumentParser(description='Recreate manuscript plots from the archived results.')
    parser.add_argument('--output', default='reproduced/figures')
    args = parser.parse_args()
    p.FIG = output_folder(args.output)
    prediction = p.read(ROOT / 'results/final_evaluation/aligned_predictions.csv')
    intervals = p.read(p.T / 'bootstrap_intervals.csv')
    metrics = p.read(p.T / 'metrics_by_dataset.csv')
    failure = p.read(p.T / 'failure_decomposition.csv')
    for extension in ['svg', 'pdf', 'png']:
        shutil.copy2(ROOT / f'publication/figures/fig01_study_design.{extension}', p.FIG)
    p.target_distributions(ROOT, prediction)
    p.performance(intervals, 'mae', 'filtered', 'fig03_mae')
    p.performance(intervals, 'spearman', 'filtered', 'fig04_spearman')
    p.calibration(prediction, 'filtered', 'fig05_calibration')
    p.shift_compression(metrics, failure)
    p.lf_and_preproc(ROOT)
    p.architecture()
    p.seeds()
    p.gdm_and_feature_shift()
    p.temporal()
    p.performance(intervals, 'mae', 'author_compatible', 'supp09_author_mae')
    p.performance(intervals, 'spearman', 'author_compatible', 'supp10_author_spearman')
    p.calibration(prediction, 'author_compatible', 'supp11_author_calibration')
    print('Created 23 numerical figure pairs and copied the editable study diagram. No models fitted.')


if __name__ == '__main__':
    main()
