from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config/study.json').read_text())
HOLDOUTS = tuple(CONFIG['holdouts'])
CONDITIONS = tuple(CONFIG['conditions'])
SSI = [f'ssi_{i:03d}' for i in range(60)]
IDENTIFIERS = ['outer_holdout', 'data_condition', 'dataset', 'pair_role', 'pair_id', 'event_uid_a', 'event_uid_b']

def read(path, round_trip=True):
    return pd.read_csv(path, keep_default_na=False, low_memory=False,
                       float_precision='round_trip' if round_trip else None)

def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def output_folder(name):
    path = (ROOT / name).resolve()
    protected = ['data_clean', 'experiments', 'results', 'audit_reports', 'analysis_plan', 'publication', 'provenance', 'analysis', 'config', '.git', 'scripts', 'original_downloads', 'release']
    if path == ROOT or not path.is_relative_to(ROOT) or any(path == ROOT / p or path.is_relative_to(ROOT / p) for p in protected):
        raise ValueError('Reproduction output must be a separate folder inside this repository.')
    path.mkdir(parents=True, exist_ok=True)
    return path

def features(round_trip=True):
    frame = read(ROOT / 'data_clean/event_features.csv', round_trip)
    if len(frame) != 8023 or frame.event_uid.duplicated().any() or not np.isfinite(frame[SSI].to_numpy(float)).all():
        raise ValueError('Invalid event feature table')
    return frame.set_index('event_uid', drop=False)

def cells(holdout='all', condition='both'):
    for h in HOLDOUTS:
        for c in CONDITIONS:
            if holdout in ('all', h) and condition in ('both', c):
                yield h, c

def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')
