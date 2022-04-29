import os

num_nodes = int(os.environ['WORLD_SIZE'])

ec = os.system(
f'''set -e;/bin/bash -c "source hfai_env openfold38;python3 -u train_fold.py ./data/processed_data/pdb_mmcif_processed ./data/processed_data/alignments ../data/pdb_mmcif/ output 2021-10-10 --template_release_dates_cache_path ./data/mmcif_cache.json --precision 32 --gpus 8 --num_nodes {num_nodes} --seed 41 --train_mapping_path ./data/data_mapping.json --use_hfai"'''
)

assert ec == 0
