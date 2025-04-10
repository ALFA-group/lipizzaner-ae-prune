import argparse
import collections
import datetime
import itertools
import json
import logging
import os
import sys
from typing import Dict, Optional
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aes_lipi.anns.prune_ann import get_n_params
from aes_lipi.utilities.utilities import (
    load_data_and_autoencoder,
    measure_ae_quality,
    measure_retrieval_error,
)


def parse_arguments(param: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse an AE ANN experiment")
    parser.add_argument(
        "--param_dir",
        type=str,
        required=True,
        help="Use directory for params",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Use directory for data",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["info", "debug", "warning"],
        default="warning",
        help="Loglevel, e.g. --log_level info",
    )
    parser.add_argument(
        "--name_map_file", type=str, default="", help="Map names to display names"
    )

    args = parser.parse_args(param)
    return args


def analyse_trials(
    root_dir: str,
    configuration_files: dict[str, str],
    name: str,
    name_map: Dict[str, str] = {},
) -> tuple[pd.DataFrame, str]:
    logging.info(f"Analyse trial data in {root_dir} with {len(configuration_files)}")
    # TODO messy
    output_dir = os.path.join(root_dir, f"experiment_analysis_{name}")
    os.makedirs(output_dir, exist_ok=True)
    dfs = []
    quality_scores = collections.defaultdict(list)
    for directory in tqdm(os.listdir(root_dir)):
        if not directory.startswith("trial_"):
            continue
        file_path = os.path.join(root_dir, directory, "stats.jsonl")
        trial_param_path = configuration_files.get(directory)
        if trial_param_path is None:
            continue

        try:
            with open(trial_param_path, "r") as fd:
                trial_params = json.load(fd)
        except ValueError as e:
            logging.error(f"{e} for loading {trial_param_path}")
            continue

        epochs = trial_params["epochs"]
        trial_nr = int(directory.split("_")[1])
        logging.info(f"Read: {file_path}")
        try:
            _df = pd.read_json(file_path, orient="records", lines=True)
        except ValueError as e:
            logging.error(f"{e} for {file_path}")
            continue

        _df["trial_param_path"] = trial_param_path
        _df["trial_nr"] = [trial_nr] * len(_df)
        dir_name = "_".join(directory.split("_")[2:])
        _df["directory"] = dir_name
        # TODO Messy display name
        display_name = get_display_name(dir_name, name_map)
        _df["display_name"] = display_name
        dfs.append(_df)
        logging.info(f"Load {file_path} {_df.shape}")
        # Measure quality
        ann_path = os.path.join(root_dir, directory, "checkpoints", f"t{epochs}_ae.pt")
        _df["final_ann_path"] = ann_path
        dataset_name = trial_params["dataset_name"]
        train_loader, test_loader, width, height, ae = load_data_and_autoencoder(
            trial_params["batch_size"],
            ann_path,
            trial_params["environment"],
            dataset_name,
            data_loader_str="test",
        )
        n_nodes = {
            "encoder": get_n_params(ae.encoder),
            "decoder": get_n_params(ae.decoder),
            "ae": get_n_params(ae.encoder) + get_n_params(ae.decoder),
        }
        try:
            quality = measure_ae_quality(
                ann_path,
                trial_params["environment"],
                trial_params["batch_size"],
                dataset_name,
            )
        except Exception as e:
            logging.error(f"{e} for {trial_param_path}")
            quality = {}

        if not quality:
            continue
        for key, value in quality.items():
            trial_id = np.multiply(np.ones(value.shape), trial_nr)
            values = np.hstack((value, trial_id))
            quality_scores[key].append(values)

    if quality_scores:
        # TODO Hacky

        key = "L1"
        qualities = np.vstack(quality_scores[key])
        columns = [key, "Trial"]
        df = pd.DataFrame(qualities, columns=columns)
        sns.boxplot(df, x=key, hue="Trial")
        out_path = os.path.join(output_dir, f"t{epochs}_{key}_bar.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.clf()
        out_path = os.path.join(output_dir, f"t{epochs}_{key}.json")
        df.to_json(out_path)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    out_file = os.path.join(output_dir, "stats.jsonl")
    exp_dir = output_dir
    df.to_json(out_file, orient="records", lines=True)
    logging.info(f"Analysed {root_dir}")
    plot_data(df, output_dir, n_nodes=n_nodes)

    return df, exp_dir


def get_display_name(display_name: str, name_map: Dict[str, str]) -> str:
    NAME_MAP = {
        "dataset_name": "",
        "binary_clustering": "bc",
        "solution_concept": "sc",
        "population_size": "ps",
        "radius": "r",
        "learning_rate": "lr",
        "epoch-node": "en",
        "True": "T",
        "False": "F",
        "environment": "env",
        "AutoencoderBinaryClustering": "AE",
        "batch_size": "bs",
        "cell_evaluation": "ce",
        "checkpoint_every_update": "",
        "10_100_1000": "10_1000",
        "ann_canonical": "can",
        "epoch-node": "en",
        "all_vs_all": "all",
        "lipi_simple": "simp",
    }
    NAME_MAP.update(name_map)
    for k, v in NAME_MAP.items():
        display_name = display_name.replace(k, v)

    if display_name.split("_")[-2].startswith("2025-"):
        display_name = "_".join(display_name.split("_")[:-2])

    return display_name


def plot_data(
    df: pd.DataFrame, output_dir: str, n_nodes: Optional[Dict[str, int]]
) -> None:
    # TODO node index is a bit arbitrary (but so is any choice)
    # Learning rates
    ordering = sorted(df["display_name"].unique())
    prune_normalizations = ("min_replacement_loss", "mean_L1")  # "test_loss")
    for ann_key in tqdm(("encoder", "decoder", "ae")):
        for performance_measure in prune_normalizations:
            col_name = f"{ann_key}_{performance_measure}_ratio"
            ann_prune_key = f"n_pruned_{ann_key}"
            if ann_key == "ae":
                n_pruned = df["n_pruned_encoder"] + df["n_pruned_decoder"]
            else:
                n_pruned = df[ann_prune_key]
            df[col_name] = df[performance_measure] / (n_nodes[ann_key] - n_pruned)
            plt.subplots(figsize=(16, 16))
            ax = sns.lineplot(
                data=df, x="iteration", y=col_name, hue="display_name", markers=True
            )
            try:
                sns.move_legend(
                    ax,
                    "lower center",
                    bbox_to_anchor=(0.5, 1),
                    ncol=1,
                    title=None,
                    frameon=False,
                )
            except ValueError as e:
                logging.error(e)

            out_path = os.path.join(output_dir, f"iteration_{col_name}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.clf()
            plt.close()

    for x_v in tqdm(
        [
            "iteration",
        ]
    ):
        plt.subplots(figsize=(16, 16))
        ax = sns.lineplot(
            data=df, x=x_v, y="learning_rate", hue="display_name", hue_order=ordering
        )
        try:
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, 1),
                ncol=1,
                title=None,
                frameon=False,
            )
        except ValueError as e:
            logging.error(e)

        out_path = os.path.join(output_dir, f"{x_v}_learning_rates.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.clf()
        # Losses
        plt.subplots(figsize=(16, 16))
        ax = sns.lineplot(
            data=df,
            x=x_v,
            y="min_replacement_loss",
            hue="display_name",
            hue_order=ordering,
        )
        try:
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, 1),
                ncol=1,
                title=None,
                frameon=False,
            )
        except ValueError as e:
            logging.error(e)

        out_path = os.path.join(output_dir, f"{x_v}_replacement_loss.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.clf()
        plt.subplots(figsize=(16, 16))
        ax = sns.lineplot(
            data=df, x=x_v, y="min_selection_loss", style="display_name", markers=True
        )
        try:
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, 1),
                ncol=1,
                title=None,
                frameon=False,
            )
        except ValueError as e:
            logging.error(e)
        out_path = os.path.join(output_dir, f"{x_v}_selection_loss.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.clf()
        plt.close()

        for ann_name in tqdm(("encoder", "decoder")):
            plt.subplots(figsize=(16, 16))
            y_value = f"n_pruned_{ann_name}"
            ax = sns.scatterplot(
                data=df, x="mean_L1", y=y_value, style="display_name", markers=True
            )
            try:
                sns.move_legend(
                    ax,
                    "lower center",
                    bbox_to_anchor=(0.5, 1),
                    ncol=1,
                    title=None,
                    frameon=False,
                )
            except ValueError as e:
                logging.error(e)
            out_path = os.path.join(output_dir, f"scatter_mean_L1_{y_value}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.clf()

            plt.subplots(figsize=(16, 16))
            ax = sns.lineplot(
                data=df,
                x="iteration",
                y=y_value,
                style="display_name",
                markers=True,
                hue="mean_L1",
            )
            try:
                sns.move_legend(
                    ax,
                    "lower center",
                    bbox_to_anchor=(0.5, 1),
                    ncol=1,
                    title=None,
                    frameon=False,
                )
            except ValueError as e:
                logging.error(e)
            out_path = os.path.join(output_dir, f"iteration_mean_L1_{y_value}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.clf()

            plt.subplots(figsize=(16, 16))
            ax = sns.lineplot(
                data=df,
                x="iteration",
                y=y_value,
                style="display_name",
                markers=True,
                hue="mean_L1",
            )
            try:
                sns.move_legend(
                    ax,
                    "lower center",
                    bbox_to_anchor=(0.5, 1),
                    ncol=1,
                    title=None,
                    frameon=False,
                )
            except ValueError as e:
                logging.error(e)
            out_path = os.path.join(output_dir, f"iteration_mean_L1_{y_value}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.clf()

            plt.subplots(figsize=(16, 16))
            ax = sns.scatterplot(
                data=df,
                x="min_replacement_loss",
                y=y_value,
                style="display_name",
                markers=True,
            )
            try:
                sns.move_legend(
                    ax,
                    "lower center",
                    bbox_to_anchor=(0.5, 1),
                    ncol=1,
                    title=None,
                    frameon=False,
                )
            except ValueError as e:
                logging.error(e)
            out_path = os.path.join(
                output_dir, f"scatter_min_replacement_loss_{y_value}.png"
            )
            plt.tight_layout()
            plt.savefig(out_path)
            plt.clf()
            plt.subplots(figsize=(16, 16))
            ax = sns.lineplot(
                data=df, x=x_v, y=y_value, style="display_name", markers=True
            )
            try:
                sns.move_legend(
                    ax,
                    "lower center",
                    bbox_to_anchor=(0.5, 1),
                    ncol=1,
                    title=None,
                    frameon=False,
                )
            except ValueError as e:
                logging.error(e)
            out_path = os.path.join(output_dir, f"{x_v}_{y_value}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.clf()

        final_gen = df[x_v].max()
        plt.subplots(figsize=(16, 16))
        ax = sns.boxplot(
            data=df[df[x_v] == final_gen],
            y="display_name",
            x="min_replacement_loss",
            order=ordering,
        )
        out_path = os.path.join(
            output_dir, f"{x_v}_t{final_gen}_min_replacement_directory_box.png"
        )
        plt.tight_layout()
        plt.savefig(out_path)
        plt.clf()

        # Quality measures
        quality_measures = [_ for _ in df.columns if _.startswith("mean_")]
        for quality_measure in tqdm(quality_measures):
            measure = "_".join(quality_measure.split("_")[1:])
            plt.clf()
            plt.subplots(figsize=(16, 16))
            ax = sns.lineplot(
                data=df, x=x_v, y=quality_measure, style="display_name", markers=True
            )
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, 1),
                ncol=1,
                title=None,
                frameon=False,
            )
            out_path = os.path.join(output_dir, f"{x_v}_{measure}_directory_line.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.clf()
            try:
                plt.subplots(figsize=(16, 16))
                ax = sns.boxplot(
                    data=df[df[x_v] == final_gen],
                    y="display_name",
                    x=quality_measure,
                    order=ordering,
                )
                out_path = os.path.join(
                    output_dir, f"{x_v}_t{final_gen}_{measure}_directory_box.png"
                )
                plt.tight_layout()
                plt.savefig(out_path)
                plt.clf()
            except ValueError as e:
                logging.error(f"{e} for {quality_measure} {x_v} {final_gen}")


def get_configuration_files(param_dir: str) -> Dict[str, str]:
    configuration_files = {}
    for root, _, files in os.walk(param_dir):
        for file_name in files:
            if file_name == "params.json":
                key = os.path.basename(root)
                configuration_files[key] = os.path.join(root, file_name)

    return configuration_files


def main(root_dir: str, param_dir: str, name_map_file: str = ""):
    configuration_files = get_configuration_files(param_dir)

    logging.info(f"Found {len(configuration_files)} cfgs in {param_dir}")
    if name_map_file != "":
        with open(name_map_file, "r") as fd:
            name_map = json.load(fd)
    else:
        name_map = {}

    try:
        df, out_dir = analyse_trials(
            root_dir, configuration_files, "GECCO_2025", name_map
        )
    except ValueError as e:
        logging.error(f"{e}")


if __name__ == "__main__":
    timestamp = "{:%Y-%m-%d_%H:%M:%S.%f}".format(datetime.datetime.now())
    log_file = os.path.basename(__file__).replace(".py", f"_{timestamp}.log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.WARNING,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    param = parse_arguments(sys.argv[1:])
    logging.getLogger().setLevel(level=param.log_level.upper())
    print(param, logging.getLogger().level, param.log_level)

    main(param.root_dir, param.param_dir, param.name_map_file)
