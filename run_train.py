import os

num_nodes = int(os.environ['WORLD_SIZE'])

ec = os.system(f'''
set -e;
/bin/bash -c "source hfai_env openfold38;cd /ceph-jd/prod/jupyter/zhangmingchuan/notebooks/yinghuo-alphafold/alphafold-optimized;/ceph-jd/prod/private/zhangmingchuan/python3 -u train_fold.py /3fs-jd/prod/platform_team/dataset/alphafold/pdb_mmcif_processed /3fs-jd/prod/platform_team/dataset/alphafold/alignments ../data/pdb_mmcif/ output 2021-10-10 --template_release_dates_cache_path ../mmcif_cache2.json --precision 32 --gpus 8 --num_nodes {num_nodes} --seed 41 --train_mapping_path ../full_dataset/final_mapping.json --use_hfai"
''')

assert ec == 0
