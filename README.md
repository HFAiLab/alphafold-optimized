# Alphafold Optimized

简体中文 | [English](README_en.md)

![model_structure](imgs/alphafold_structure.png)

本项目实现了一个性能优化版的Alphafold，大幅加速了Alphafold模型的训练。基于[Openfold](https://github.com/aqlaboratory/openfold)实现的Pytorch版Alphafold进行优化，通过对数据处理、读取的优化以及hfai.nn提供的算子优化提升了模型的整体训练性能。

## 使用方式

### 环境要求

    python>=3.8
    pip install -r requirements.txt

### 数据预处理
Alphafold Optimized使用的是已经预处理后的数据集，可以从OSS下载。

### 模型训练

Alphafold模型的训练对显存的要求较高，至少需要22GB的GPU显存进行训练。

提交任务至萤火集群：

```shell
hfai python run_train.py -- -n 4 -p 30
```

本地运行：

```shell
source hfai_env openfold38
python train_fold.py /3fs-jd/prod/platform_team/dataset/alphafold/pdb_mmcif_processed /3fs-jd/prod/platform_team/dataset/alphafold/alignments ../data/pdb_mmcif/ output 2021-10-10 --template_release_dates_cache_path ../mmcif_cache2.json --precision 32 --gpus 1 --num_nodes 1 --seed 41 --train_mapping_path ../full_dataset/final_mapping.json --use_hfai
```

### 模型推理

```shell
source hfai_env openfold38
python run_pretrained_openfold.py example_data/fasta/1ak0_1_A.fasta \
    data/uniref90/uniref90.fasta \
    data/mgnify/mgy_clusters_2018_12.fa \
    data/pdb70/pdb70 \
    data/pdb_mmcif/ \
    data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --output_dir ./ \
    --bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --model_device cuda:1 \
    --jackhmmer_binary_path /ceph-jd/pub/marsV2/lib/conda/envs/openfold_venv/bin/jackhmmer \
    --hhblits_binary_path /ceph-jd/pub/marsV2/lib/conda/envs/openfold_venv/bin/hhblits \
    --hhsearch_binary_path /ceph-jd/pub/marsV2/lib/conda/envs/openfold_venv/bin/hhsearch \
    --kalign_binary_path /ceph-jd/pub/marsV2/lib/conda/envs/openfold_venv/bin/kalign

```
