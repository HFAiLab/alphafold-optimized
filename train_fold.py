import argparse
import os
import logging

import pytorch_lightning as pl
import torch
from hfai.nn import to_hfai
import hfai.nn.parallel.DistributedDataParallel as hfai_DDP
import torch.nn.parallel.DistributedDataParallel as torch_DDP

from openfold.config import model_config
from openfold.data.data_modules import (
    OpenFoldDataModule,
)
from openfold.model.model import AlphaFold
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.argparse import remove_arguments
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.seed import seed_everything
from openfold.utils.tensor_utils import tensor_tree_map    

def main(local_rank, args):
    if(args.seed is not None):
        seed_everything(args.seed)
    if args.use_hfai:
        DDP = hfai_DDP
    else:
        DDP = torch_DDP
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    
    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://{}:{}'.format(ip, port), world_size=hosts*args.gpus, rank=rank*args.gpus+local_rank)    
    
    torch.cuda.set_device(local_rank)

    config = model_config(
        "model_1", 
        train=True, 
        low_prec=(args.precision == 16)
    ) 
    
    device = torch.device("cuda", 0)
    model = AlphaFold(config).cuda()
    if args.use_hfai:
        model = to_hfai(model)
    model = DDP(model, device_ids=[local_rank])
    
    loss_func = AlphaFoldLoss(config.loss).cuda()
    modelEMA = ExponentialMovingAverage(
        model=model,
        decay=config.ema.decay
    )
    modelEMA.to(device)
    learning_rate = 1e-3
    eps = 1e-8
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        eps=eps
    )

    log = logging.getLogger()
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    if local_rank == 0:
        log.info(f'Preparing training data')
    
    data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )
    data_module.prepare_data()
    data_module.setup()
    
    
    epoch_num = 10
    for epoch in range(epoch_num):
        if local_rank == 0:
            log.info(f'Epoch {epoch} started')
            batch_loss = []

        dataloader = data_module._gen_dataloader("train", replace_ddp_sampler=True)
        for idx, batch in enumerate(dataloader):
            for feature in batch:
                batch[feature] = batch[feature].cuda()
            model.zero_grad()
            output = model(batch)
            target = tensor_tree_map(lambda t: t[..., -1], batch)
            modelEMA.update(model)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            
            if local_rank == 0:
                batch_loss.append(loss.item())
                log.info(f'Epoch: {epoch} Step: {idx+1}/{len(dataloader)} Loss: {round(sum(batch_loss)/len(batch_loss), 4)}')
                    


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "train_alignment_dir", type=str,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "template_mmcif_dir", type=str,
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_mapping_path", type=str, default=None,
        help='''Optional path to a .json file containing a mapping from
                consecutive numerical indices to sample names. Used to filter
                the training set'''
    )
    parser.add_argument(
        "--distillation_mapping_path", type=str, default=None,
        help="""See --train_mapping_path"""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_best_val", type=bool_type, default=True,
        help="""Whether to save the model parameters that perform best during
                validation"""
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--log_performance", type=bool_type, default=False,
        help="Measure performance"
    )
    parser.add_argument(
        "--script_modules", type=bool_type, default=False,
        help="Whether to TorchScript eligible components of them model"
    )
    parser.add_argument(
        "--use_hfai", type=bool_type, default=True,
        help="Whether to use hfai.nn's optimized DDP and primitives or not"
    )
    parser = pl.Trainer.add_argparse_args(parser)
   
    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )

    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(parser, ["--accelerator", "--resume_from_checkpoint"]) 

    args = parser.parse_args()

    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    #main(args)
    torch.multiprocessing.spawn(main, args=[args], nprocs=args.gpus)
