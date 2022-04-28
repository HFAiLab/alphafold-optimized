import argparse
import logging
import numpy as np
import os

os.environ["OPENMM_DEFAULT_PLATFORM"] = "CUDA"

import random
import sys
import time
import torch

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.model.model import AlphaFold
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map

from scripts.utils import add_data_args


def main(args):
    config = model_config("model_1", train=False) 
    device = torch.device("cuda", 1)
    model = AlphaFold(config).to(device)
    model = model.eval()
    # import_jax_weights_(model, args.param_path, version=args.model_name)
    if "ema" in args.param_path:
        model.load_state_dict(torch.load(args.param_path)["params"])
    else:
        model.load_state_dict(torch.load(args.param_path))
    
 
    template_featurizer = templates.TemplateHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=config.data.predict.max_templates,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=args.obsolete_pdbs_path
    )

    use_small_bfd=(args.bfd_database_path is None)

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)
    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if(args.use_precomputed_alignments is None):
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    # Gather input sequences
    with open(args.fasta_path, "r") as fp:
        lines = [l.strip() for l in fp.readlines()]

    tags, seqs = lines[::2], lines[1::2]
    tags = [l[1:] for l in tags]

    for tag, seq in zip(tags, seqs):
        fasta_path = os.path.join(args.output_dir, "tmp.fasta")
        with open(fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        logging.info("Generating features...") 
        local_alignment_dir = os.path.join(alignment_dir, tag)
        if(args.use_precomputed_alignments is None):
            if not os.path.exists(local_alignment_dir):
                os.makedirs(local_alignment_dir)
            
            alignment_runner = data_pipeline.MTAlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                small_bfd_database_path=args.small_bfd_database_path,
                pdb70_database_path=args.pdb70_database_path,
                use_small_bfd=use_small_bfd,
                no_cpus=args.cpus,
            )
            alignment_runner.run(
                fasta_path, local_alignment_dir
            )
    
        feature_dict = data_processor.process_fasta(
            fasta_path=fasta_path, alignment_dir=local_alignment_dir
        )

        # Remove temporary FASTA file
        os.remove(fasta_path)
    
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict',
        )
    
        logging.info("Executing model...")
        batch = processed_feature_dict
        with torch.no_grad():
            batch = {
                k:torch.as_tensor(v, device=args.model_device) 
                for k,v in batch.items()
            }
        
            t = time.perf_counter()
            out = model(batch)
            logging.info(f"Inference time: {time.perf_counter() - t}")
       
        # Toss out the recycling dimensions --- we don't need them anymore
        batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
        
        plddt = out["plddt"]
        mean_plddt = np.mean(plddt)
        
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
    
        unrelaxed_protein = protein.from_prediction(
            features=batch,
            result=out,
            b_factors=plddt_b_factors
        )
         
        amber_relaxer = relax.AmberRelaxation(
            **config.relax
        )
        
        # Relax the prediction.
        t = time.perf_counter()
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
        logging.info(f"Relaxation time: {time.perf_counter() - t}")
        
        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(
            args.output_dir, f'{tag}_{args.model_name}.pdb'
        )
        with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_path", type=str,
    )
    add_data_args(parser)
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
        required=True
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--model_name", type=str, default="model_1",
        help="""Name of a model config. Choose one of model_{1-5} or 
             model_{1-5}_ptm, as defined on the AlphaFold GitHub."""
    )
    parser.add_argument(
        "--param_path", type=str, default=None,
        help="""Path to model parameters. If None, parameters are selected
             automatically according to the model name from 
             openfold/resources/params"""
    )
    parser.add_argument(
        "--cpus", type=int, default=100,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        '--preset', type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        '--data_random_seed', type=str, default=None
    )

    args = parser.parse_args()

    if(args.param_path is None):
        args.param_path = os.path.join(
            "openfold", "resources", "params", 
            "params_" + args.model_name + ".npz"
        )

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    if(args.bfd_database_path is None and 
       args.small_bfd_database_path is None):
        raise ValueError(
            "At least one of --bfd_database_path or --small_bfd_database_path"
            "must be specified"
        )

    main(args)
