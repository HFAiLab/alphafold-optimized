import os

num_nodes = int(os.environ['WORLD_SIZE'])

ec = os.system(
f'''set -e;/bin/bash -c "python3 -u train_fold.py . . /public_dataset/1/ffdataset/Alphafold/pdb_mmcif/ output 2021-10-10 --template_release_dates_cache_path /public_dataset/1/ffdataset/Alphafold/mmcif_cache_all.json --precision 32 --gpus 8 --num_nodes {num_nodes} --seed 41 --train_mapping_path . --use_hfai"'''
)

assert ec == 0
