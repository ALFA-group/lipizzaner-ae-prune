import datetime
import json
import os
import sys
import argparse
import logging
from typing import List, Tuple

import torch

from aes_lipi.algorithms.canonical_ae_training import evaluate_ann_canonical
from aes_lipi.algorithms.lipi_ae_base import (
    best_case,
    checkpoint_ae,
    get_best_nodes,
    device,
    initialize_nodes,
)
from aes_lipi.algorithms.lipi_ae_update_only_selected import evaluate_cells_lipi_simple
from aes_lipi.algorithms.lipi_ae_update_only_selected_prune_then_eval import evaluate_cells_lipi_simple_prune_then_eval
from aes_lipi.algorithms.lipi_ae_update_selected_and_random import (
    evaluate_cells_all_vs_all,
    evaluate_cells_epochs_nodes,
)
from aes_lipi.datasets.data_loader import create_batches
from aes_lipi.utilities.utilities import (
    plot_stats,
    set_rng_seed,
    show_images,
    str2bool,
)


def parse_arguments(param: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments (`sys.argv`).
    """
    parser = argparse.ArgumentParser(description="Run AE ANN")
    parser.add_argument(
        "--configuration_file",
        type=str,
        help="JSON configuration file. E.g. configurations/demo_lipi_ae.json",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default="none",
        choices=("all", "final", "none"),
        help="Visualize ANNs. E.g. all",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset that will be used. E.g. mnist",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10000,
        help="Checkpoint ANN interval. E.g. 10",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        help="Rng seed. E.g. 10",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="Neighborhood radius. E.g. 1",
    )
    parser.add_argument(
        "--ann_path",
        type=str,
        default="",
        help="Path to load ANNs. IDX in path will be replaced by node index. E.g. checkpoints/nIDX_ae.pt",
    )
    parser.add_argument(
        "--cell_evaluation",
        type=str,
        default="epoch-node",
        choices=("epoch-node", "all_vs_all", "ann_canonical", "lipi_simple", "lipi_simple_pte"),
        help="Evaluation of cells. E.g. epoch-node",
    )
    parser.add_argument(
        "--ae_quality_measures",
        type=str,
        default="all",
        help="Comma separated list of ae quality measures. 'all' means all measures. E.g. FID,SSIM",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out_data",
        help="Output directory. Timestamp is appended. E.g. mlp_img",
    )
    parser.add_argument(
        "--environment",
        type=str,
        help="Environment as instanciated from utilities. E.g. AutoencoderBinaryClustering",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["info", "debug", "warning"],
        default="info",
        help="Loglevel, e.g. --log_level info",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=2,
        help="Population size. E.g. 2",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs. E.g. 2",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4000,
        help="Batch size. E.g. 2",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate. E.g. 0.0005",
    )
    parser.add_argument(
        "--prune_probability",
        type=float,
        default=0.0,
        help="Probability of pruning a network after selection. E.g. 0.5",
    )
    parser.add_argument(
        "--prune_amount",
        type=float,
        default=0.1,
        help="Amount to prune. E.g. 0.1",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=["None", "activation", "random"],
        default="None",
        help="Method for pruning, e.g. --prune_method random",
    )
    parser.add_argument(
        "--prune_schedule",
        type=str,
        choices=["fixed", "increase", "population"],
        default="fixed",
        help="Schedule for pruning probability, e.g. --prune_method increase",
    )
    parser.add_argument(
        "--no_execution",
        action="store_true",
        help="Do not run. Used for testing",
    )
    parser.add_argument(
        "--do_not_overwrite_checkpoint",
        action="store_false",
        default=True,
        help="Do not overwrite checkpoint",
        dest="overwrite_checkpoint",
    )
    parser.add_argument(
        "--store_all_nodes",
        action="store_true",
        default=False,
        help="Store all nodes. Useful for creating ensembles",
    )
    parser.add_argument(
        "--calculate_test_loss",
        type=str2bool,
        default=False,
        help="Calculate test set loss each epoch",
    )
    parser.add_argument(
        "--checkpoint_every_update",
        type=str2bool,
        default=False,
        help="Checkpoint ANN after each update",
    )
    parser.add_argument(
        "--no_shuffle_data",
        type=str2bool,
        default=False,
        help="Do not shuffle data in data loader",
    )
    # TODO HACK
    parser.add_argument(
        "--no_output_dir_timestamp",
        type=str2bool,
        default=False,
        help="Do not add timestamp to output folder",
    )

    args = parser.parse_args(param)
    return args


def main(
    population_size: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    visualize: str,
    cell_evaluation: str,
    environment: str,
    dataset_name: str,
    output_dir: str,
    checkpoint_interval: int,
    radius: int,
    **kwargs,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    os.makedirs(output_dir, exist_ok=True)

    kwargs["original_learning_rate"] = learning_rate
    rng_seed = kwargs.get("rng_seed", None)
    solution_concept = best_case

    score_keys = [
        _.strip() for _ in kwargs.get("ae_quality_measures", "all").split(",")
    ]
    logging.info(
        f"Run with pop size:{population_size} epochs:{epochs} lr:{learning_rate} bs:{batch_size} {rng_seed} using {device}"
    )
    stats = []
    # TODO store the used seed in the params
    set_rng_seed(rng_seed)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    # Create Batches
    # TODO messy no_shuffle data negation
    training_data, test_data, width, height = create_batches(
        batch_size,
        dataset_name,
        shuffle_training=not kwargs.get("no_shuffle_data", False),
    )
    if (
        kwargs.get("calculate_test_loss", False)
        or kwargs.get("prune_method", False) == "lexicase"
    ):
        kwargs["test_data"] = test_data

    if kwargs.get("no_shuffle_data", False):
        # TODO hack for checking random
        assert training_data.sampler.__class__.__name__ == "SequentialSampler"

    # Create graph
    nodes = initialize_nodes(
        learning_rate,
        population_size,
        environment,
        kwargs.get("ann_path", ""),
        training_data,
        width,
        height,
        dataset_fraction=kwargs.get("data_subsample", 0.0),
        identical_initial_ann=kwargs.get("identical_initial_ann", False),
    )
    if cell_evaluation == "epoch-node":
        evaluate_cells_epochs_nodes(
            nodes,
            epochs,
            visualize,
            stats,
            output_dir,
            checkpoint_interval,
            radius,
            score_keys,
            solution_concept,
            **kwargs,
        )
    elif cell_evaluation == "all_vs_all":
        nodes = evaluate_cells_all_vs_all(
            nodes,
            epochs,
            visualize,
            stats,
            output_dir,
            checkpoint_interval,
            score_keys,
            solution_concept,
            training_data,
            **kwargs,
        )
    elif cell_evaluation == "lipi_simple":
        evaluate_cells_lipi_simple(
            nodes,
            epochs,
            visualize,
            stats,
            output_dir,
            checkpoint_interval,
            radius,
            score_keys,
            solution_concept,
            **kwargs,
        )
    elif cell_evaluation == "lipi_simple_pte":
        evaluate_cells_lipi_simple_prune_then_eval(
            nodes,
            epochs,
            visualize,
            stats,
            output_dir,
            checkpoint_interval,
            radius,
            score_keys,
            solution_concept,
            **kwargs,
        )
    elif cell_evaluation == "ann_canonical":
        assert population_size == 1
        nodes = evaluate_ann_canonical(
            nodes,
            epochs,
            visualize,
            stats,
            output_dir,
            checkpoint_interval,
            score_keys,
            **kwargs,
        )
    elif cell_evaluation == "async":
        raise NotImplementedError("async is not implemented")
    else:
        raise NotImplementedError(f"{cell_evaluation} is not implemented")

    best_encoder, best_decoder, idx = get_best_nodes(
        list(nodes.values()), test_data, visualize, epochs, output_dir
    )
    if visualize != "none":
        show_images(best_decoder, output_dir)

    # print("Stats", stats)
    plot_stats(stats, output_dir, id=kwargs["timestamp"])
    out_path = os.path.join(output_dir, "checkpoints", f"t{epochs}_ae.pt")
    data = {"encoder": best_encoder.state_dict(), "decoder": best_decoder.state_dict()}
    torch.save(data, out_path)
    logging.debug(f"Save best ANNs from {idx} to {out_path}")
    if kwargs.get("store_all_nodes", False):
        logging.info("Save all node centers")
        for node in nodes.values():
            ensemble_path = os.path.join(output_dir, "checkpoints", "ensemble")
            os.makedirs(ensemble_path, exist_ok=True)
            checkpoint_ae(
                output_dir=ensemble_path,
                encoder=node.encoders[0],
                decoder=node.decoders[0],
                node_idx=node.index,
                t=epochs,
                overwrite_checkpoint=False,
            )

    return best_encoder, best_decoder


if __name__ == "__main__":
    _timestamp = datetime.datetime.now()
    timestamp_str = "{:%Y-%m-%d_%H:%M:%S.%f}".format(_timestamp)
    log_file = os.path.basename(__file__).replace(".py", f"_{timestamp_str}.log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file)],
    )
    param = parse_arguments(sys.argv[1:])
    logging.getLogger().setLevel(level=param.log_level.upper())
    print(logging.getLogger().level, param.log_level)
    # TODO messy, how about overriding from the command line?
    if param.configuration_file:
        logging.info(
            f"Loading param from file {param.configuration_file} all specified command-line arguments are ignored"
        )
        with open(param.configuration_file, "r") as fd:
            params = json.load(fd)
    else:
        params = vars(param)

    params["timestamp"] = datetime.datetime.timestamp(_timestamp)
    if params["no_output_dir_timestamp"]:
        params["output_dir"] = f"{params['output_dir']}"
    else:
        params["output_dir"] = f"{params['output_dir']}_{timestamp_str}"

    logging.info(f"Using params {params}")
    # TODO ugly
    os.makedirs(params["output_dir"], exist_ok=True)
    _out_path = os.path.join(params["output_dir"], "params.json")
    with open(_out_path, "w") as fd:
        json.dump(params, fd, indent=1)

    if param.no_execution:
        logging.info("No execution flag")
        sys.exit(0)

    try:
        main(**params)
    except RuntimeError as e:
        logging.error(e)
