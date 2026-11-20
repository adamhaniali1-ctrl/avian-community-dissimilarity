import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
import yaml
from analysis.common import ROOT, read, features, cells, output_folder, checksum
from analysis import manifests


def verify_files():
    records = json.loads((ROOT / 'provenance/data_checksums.json').read_text())
    for name, expected in records.items():
        path = ROOT / name
        if not path.is_file() or checksum(path) != expected:
            raise ValueError(f'Missing or changed file: {name}')
    return len(records)


def regenerate(out):
    config = yaml.safe_load((ROOT / 'config/experiment.yaml').read_text())
    frame = features().reset_index(drop=True)
    labels = read(ROOT / 'data_clean/event_species.csv').groupby('event_uid').species_name.agg(frozenset).to_dict()
    original = {uid: labels.get(uid, frozenset()) for uid in frame.event_uid}
    base = manifests.create_base_split(frame, config)
    count = 0

    def save(table, relative):
        nonlocal count
        text = manifests.dataframe_text(table)
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if path.read_bytes() != (ROOT / relative).read_bytes():
            raise ValueError(f'Regenerated manifest differs: {relative}')
        count += 1

    save(base, 'experiments/base_event_split.csv')
    for h, c in cells():
        communities, eligible, reasons, removed = manifests.condition_species_sets(original, c)
        roles = manifests.create_event_roles(frame, base, original, communities, eligible, reasons, removed, h, c, config['datasets'], config['random_seed'])
        target = min(manifests.pair_count(sum(eligible[uid] for uid in frame.loc[frame.dataset == d, 'event_uid'])) for d in config['datasets'])
        outer, inner, counts = manifests.build_pair_manifests(roles, communities, h, c, config['datasets'], config['random_seed'], target)
        folder = f'experiments/holdout_{h}/{c}'
        for name, table in [('event_roles', roles), ('outer_pairs', outer), ('inner_pairs', inner), ('pair_counts', counts)]:
            save(table, f'{folder}/{name}.csv')
    return count


def main():
    parser = argparse.ArgumentParser(description='Download the review data or reproduce the event and pair manifests.')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--regenerate-manifests', action='store_true')
    parser.add_argument('--output', default='reproduced')
    args = parser.parse_args()
    archive = args.archive
    if args.download:
        archive = ROOT / 'review_data_v1.0.0.zip'
        urlretrieve('https://github.com/adamhaniali1-ctrl/avian-community-dissimilarity/releases/download/v1.0.0/review_data_v1.0.0.zip', archive)
    if archive:
        with zipfile.ZipFile(archive) as source:
            allowed = json.loads((ROOT / 'provenance/data_checksums.json').read_text())
            if set(source.namelist()) != set(allowed):
                raise ValueError('Archive contents do not match the published file list')
            for name in source.namelist():
                destination = (ROOT / name).resolve()
                if not destination.is_relative_to(ROOT) or destination == ROOT:
                    raise ValueError('Unsafe archive path')
                data = source.read(name)
                import hashlib
                if hashlib.sha256(data).hexdigest() != allowed[name]:
                    raise ValueError(f'Archive checksum mismatch: {name}')
                if destination.exists() and checksum(destination) != allowed[name]:
                    raise ValueError(f'Extraction would replace a changed file: {name}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
    count = verify_files()
    print(f'PASS: {count} archived files match their SHA256 checksums.')
    if args.regenerate_manifests:
        count = regenerate(output_folder(args.output))
        print(f'PASS: {count} regenerated manifests are byte identical to the research record.')


if __name__ == '__main__':
    main()
