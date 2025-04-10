import argparse
import copy
import datetime
import itertools
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from tqdm import tqdm

sys.path.append("src/")

from aes_lipi import lipi_ae
from aes_lipi.datasets.data_loader import create_binary_cluster_data
from aes_lipi.utilities import analyse_data
from aes_lipi.utilities.gecco_experiments import is_blacklisted
from aes_lipi.utilities.utilities import get_lipiae_command_line_command, str2bool

experiment_configuration_defaults = {
    "epochs": 400,
    # Fitness Evaluations?
    "learning_rate": 1e-05,
    "visualize": "final",
    "trials": 30,
    "checkpoint_interval": 10000,
    "rng_seed": 1,
    "radius": 1,
    "log_level": "debug",
    "ae_quality_measures": "L1",
    "calculate_test_loss": True,
    "output_dir": "out_bc_aa_gecco_25",
    "batch_size": 5,
    "lexi_threshold": 0.1,
}
rq_sensitivities = {
    "capacity": {
        "cell_evaluation": ["lipi_simple_pte"],
        "population_size": [10],
        "epochs": [4],
        "trials": [1],
        "environment": [
            "AutoencoderBinaryClustering",
            # "AutoencoderBinaryClustering_Small",
        ],
        "dataset_name": ["binary_clustering_10_100_1000"],
        "prune_method": ["random", "lexicase"],
        "prune_schedule": [
            "exponential",
        ],
        "prune_probability": [0.5],
        "prune_amount": [0.1],
    },
    "cell_evaluations": {
        "cell_evaluation": ["epoch-node", "ann_canonical", "all_vs_all", "lipi_simple"],
        "population_size": [5],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": ["binary_clustering_10_100_1000"],
    },
    "population_size": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [1, 2, 5, 20],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": ["binary_clustering_10_100_1000"],
    },
    "environment": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [5],
        "environment": [
            "AutoencoderBinaryClustering",
            "DenoisingAutoencoderBinaryClustering",
            "VariationalAutoencoderBinaryClustering",
            "AutoencoderBinaryClustering_Small",
            "AutoencoderBinaryClustering_Large",
        ],
        "dataset_name": ["binary_clustering_10_100_1000"],
    },
    "dataset_name": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [10],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": [
            "binary_clustering_50_100_4000",
        ],
        "prune_method": ["lexicase"],
        "prune_schedule": [
            "exponential",
            "final_n",
        ],
        "prune_probability": [0.5],
        "prune_amount": [0.1],
    },
    "base_pruning": {
        "cell_evaluation": ["ann_canonical"],
        "population_size": [10],
        "epochs": [4000],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": [
            "binary_clustering_10_100_1000",
        ],
        "prune_method": ["random", "lexicase"],
        "prune_schedule": [
            "fixed",
            "increase",
            "population",
            "decrease",
            "exponential",
            "final_n",
        ],
        "prune_probability": [0.5],
        "prune_amount": [0.1],
    },
    "activation_sched": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [10],
        "epochs": [400],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": [
            "binary_clustering_10_100_1000",
        ],
        "prune_method": ["activation"],
        "prune_schedule": [
            "fixed",
            "increase",
            "population",
            "decrease",
            "exponential",
            "final_n",
        ],
        "prune_probability": [0.1],
        "prune_amount": [0.1],
    },
    "best_base_activation": {
        "cell_evaluation": ["ann_canonical"],
        "population_size": [10],
        "epochs": [400],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": [
            "binary_clustering_10_100_1000",
        ],
        "prune_method": ["activation"],
        "prune_schedule": [
            # "fixed",
            # "increase",
            # "population",
            # "decrease",
            "exponential",
            # "final_n",
        ],
        "prune_probability": [0.1],
        "prune_amount": [0.1],
    },
    "lexi_prob": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [5],
        "epochs": [400],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": [
            "binary_clustering_10_100_1000",
        ],
        "prune_method": ["random", "lexicase"],
        "prune_schedule": [
            "fixed",
            "increase",
            "population",
            "decrease",
            "exponential",
            "final_n",
        ],
        "prune_probability": [0.5],
        "prune_amount": [0.1],
    },
    "prune_after": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [10],
        "prune_method": ["lexicase"],
        "prune_schedule": [
            # "fixed",
            # "increase",
            # "population",
            # "decrease",
            # "exponential",
            "final_n",
        ],
        "prune_probability": [0.1, 0.9],
        "prune_amount": [0.1],
    },
    "lipi_small_4000": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [10],
        "epochs": [400],
        "environment": [
            "AutoencoderBinaryClustering_Small",
        ],
        "dataset_name": [
            "binary_clustering_10_100_4000",
        ],
        "prune_method": ["random"],
        "prune_schedule": [
            "exponential"
        ],
        "prune_probability": [0.5],
        "prune_amount": [0.1],
    },
    "lexi_threshold": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [5],
        "epochs": [400],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": [
            "binary_clustering_10_100_1000",
        ],
        "prune_method": ["random", "lexicase", "activation"],
        "prune_schedule": [
            "prune_after",
        ],
        "prune_probability": [1.0],
        "prune_amount": [0.1],
    },
    "act_pop_small": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [10],
        "epochs": [400],
        "environment": [
            "AutoencoderBinaryClustering_Small",
        ],
        "dataset_name": [
            "binary_clustering_10_100_1000",
        ],
        "prune_method": ["activation"],
        "prune_schedule": [
            "population"
        ],
        "prune_probability": [0.1],
        "prune_amount": [0.1],
    },
    "lex_final_n_small": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [10],
        "epochs": [400],
        "environment": [
            "AutoencoderBinaryClustering_Small",
        ],
        "dataset_name": [
            "binary_clustering_10_100_1000",
        ],
        "prune_method": ["lexicase"],
        "prune_schedule": [
            "final_n"
        ],
        "prune_probability": [0.5],
        "prune_amount": [0.1],
    },
    "prune_after_base": {
        "cell_evaluation": ["ann_canonical"],
        "population_size": [10],
        "epochs": [400],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": [
            "binary_clustering_10_100_1000",
        ],
        "prune_method": ["random", "lexicase", "activation"],
        "prune_schedule": [
            "prune_after",
        ],
        "prune_probability": [1.0],
        "prune_amount": [0.1],
    },
    "no_shuffle_data": {
        "cell_evaluation": ["lipi_simple"],
        "population_size": [5],
        "environment": [
            "AutoencoderBinaryClustering",
        ],
        "dataset_name": [
            "binary_clustering_10_100_1000",
        ],
        "prune_method": ["None", "activation"],
        "prune_schedule": ["fixed"],
        "prune_probability": [0.5],
        "prune_amount": [0.1],
        "no_shuffle_data": [True, False],
    },
}


def main(
    timestamp: datetime = "NA",
    params: Optional[Dict[str, Any]] = None,
    sensitivities: Optional[Dict[str, List[Any]]] = None,
    test: bool = False,
    print_cli_only: bool = True,
    variant_id: int = -1,
    trial_id: int = -1
) -> Dict[str, Any]:
    params["timestamp"] = timestamp
    output_dir = params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    products = list(itertools.product(*sensitivities.values()))
    variants = []
    for product in products:
        variants.append(dict((k, v) for k, v in zip(sensitivities.keys(), product)))

    if test:
        params["epochs"] = 4
        params["trials"] = 2

    cli_cmds = set()
    print("Number of Variants:", len(variants))
    print("trial_id", trial_id)
    for i, variant in tqdm(enumerate(variants), position=0, leave=True, desc="Variant"):
        if variant_id != -1:
            if variant_id != i:
                continue
        
        trials = variant.get("trials", params["trials"])
        if test:
            trials = 2

        # if variant.get("prune_method", "") == "lexicase":
        #     if variant.get("prune_schedule", "") != "fixed":
        #         logging.info(f"Skipping lexicase variant: {variant}")
        #         continue

        logging.info(f"variant: {variant} for {trials}")
        trial_params = copy.deepcopy(params)
        trial_params.update(variant)
        for i in range(trials):
            if trial_id != -1:
                if trial_id != i:
                    continue
            logging.info(f"Trial {i}")
            trial_params = copy.deepcopy(params)
            trial_params.update(variant)
            if trial_params.get("cell_evaluation", "") == "ann_canonical":
                trial_params["population_size"] = 1

            for k, v in trial_params.items():
                if v in ("True", "False"):
                    trial_params[k] = str2bool(v)
                    assert isinstance(trial_params[k], bool)

            # epochs = params.get("epochs")

            variant_name = "_".join(
                ["_".join([str(k[:2]), str(v)]) for k, v in variant.items()]
            )
            _time_stamp = datetime.datetime.now()
            trial_params["output_dir"] = os.path.join(
                output_dir, f"trial_{i}_{variant_name}"
            )
            trial_params["rng_seed"] = (
                params.get("rng_seed", int(datetime.datetime.timestamp(_time_stamp)))
                + i
            )
            # trial_params["epochs"] = epochs
            timestamp_str = "{:%Y-%m-%d_%H:%M:%S.%f}".format(_time_stamp)
            trial_params["output_dir"] = f"{trial_params['output_dir']}_{timestamp_str}"

            os.makedirs(trial_params["output_dir"], exist_ok=True)
            out_path = os.path.join(trial_params["output_dir"], "params.json")
            with open(out_path, "w") as fd:
                json.dump(trial_params, fd, indent=1)

            try:
                logging.info(f"Running experiment with {trial_params} at {timestamp}")
                cli_cmd = get_lipiae_command_line_command(trial_params)
                if print_cli_only and cli_cmd not in cli_cmds:
                    print(cli_cmd)
                    continue
                else:
                    tqdm.write(f"CLI Command:\n{cli_cmd}")
                cli_cmds.add(cli_cmd)
                # TODO no execution flag
                logging.info(f"CLI Command:\n{cli_cmd}")
                lipi_ae.main(**trial_params)
            except Exception as e:
                logging.error(f"{e} PLEASE RERUN\n{trial_params}")
                raise Exception(e)


def create_data() -> None:
    """Create datasets for the experiments."""
    variants = (
        {
            "n_dim": 1000,
            "n_clusters": 10,
            "n_examples_per_cluster": 100,
            "noise_probability": 0.05,
        },
        {
            "n_dim": 1000,
            "n_clusters": 50,
            "n_examples_per_cluster": 100,
            "noise_probability": 0.05,
        },
        {
            "n_dim": 4000,
            "n_clusters": 10,
            "n_examples_per_cluster": 100,
            "noise_probability": 0.05,
        },
        {
            "n_dim": 4000,
            "n_clusters": 50,
            "n_examples_per_cluster": 100,
            "noise_probability": 0.05,
        },
    )
    for variant in variants:
        create_binary_cluster_data(**variant)


def parse_arguments(param: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments (`sys.argv`).
    """
    parser = argparse.ArgumentParser(description="Run AE ANN experiment for GECCO 25")
    parser.add_argument("--test", type=str2bool, default=False, help="Test settings")
    parser.add_argument(
        "--print_cli_only",
        type=str2bool,
        default=False,
        help="Only print the CLI command. Do not run the command",
    )
    parser.add_argument(
        "--analyse_data", type=str2bool, default=False, help="Analyse data"
    )
    parser.add_argument(
        "--variant_id", type=int, default=-1, help="which variant to run (for supercomputer). -1 ignores this arg"
    )
    parser.add_argument(
        "--trial_id", type=int, default=-1, help="which trial to run (for supercomputer). -1 ignores this arg"
    )
    parser.add_argument(
        "--rq",
        type=str,
        choices=rq_sensitivities.keys(),
        required=True,
        default="test",
        help="Sensitivity settings for the research question, e.g. --rq population_size",
    )
    parser.add_argument(
        "--create_data", type=str2bool, default=False, help="Create the datasets"
    )
    parser.add_argument(
        "--trial_id",
        type=int,
        default=-1,
        help="Which trial to run (value of -1 implies all trials)",
    )
    parser.add_argument(
        "--all_rqs", type=str2bool, default=False, help="Run all rq settings"
    )

    args = parser.parse_args(param)
    return args


if __name__ == "__main__":
    timestamp = "{:%Y-%m-%d_%H:%M:%S.%f}".format(datetime.datetime.now())
    log_file = os.path.basename(__file__).replace(".py", f"_{timestamp}.log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler("./logs/" + log_file)],
    )

    args = parse_arguments(sys.argv[1:])
    if args.create_data:
        create_data()
        sys.exit(0)
    if args.all_rqs:
        for key, sensitivity_values in rq_sensitivities.items():
            if key == "test":
                continue

            sensitivity_values = rq_sensitivities[key]
            output_dir = f"out_{key}_gecco_25"
            experiment_configuration_defaults["output_dir"] = output_dir
            main(
                timestamp=timestamp,
                params=experiment_configuration_defaults,
                sensitivities=sensitivity_values,
                test=args.test,
                print_cli_only=args.print_cli_only,
                trial_id=args.trial_id,
            )

            analyse_data.main(
                root_dir=output_dir, param_dir=output_dir, name_map_file=""
            )

        sys.exit(0)

    sensitivity_values = rq_sensitivities[args.rq]
    output_dir = f"out_{args.rq}_gecco_25"
    experiment_configuration_defaults["output_dir"] = output_dir
    main(
        timestamp=timestamp,
        params=experiment_configuration_defaults,
        sensitivities=sensitivity_values,
        test=args.test,
        print_cli_only=args.print_cli_only,
        variant_id=args.variant_id,
        trial_id=args.trial_id
    )

    if args.analyse_data:
        analyse_data.main(root_dir=output_dir, param_dir=output_dir, name_map_file="")
