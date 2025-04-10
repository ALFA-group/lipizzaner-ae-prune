import argparse
import collections
import itertools
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns

from aes_lipi.utilities.utilities import l2_norm_diff, load_data_and_autoencoder


INPUT_LAYER_NAME = "Input"


def get_weights(ann) -> np.ndarray:
    # TODO get shape and pre allocate?
    weights = None
    for name, _ in ann.named_parameters():
        params = ann.get_parameter(name)
        if params.requires_grad:
            layer_weights = torch.flatten(params).detach().numpy()
            if weights is None:
                weights = layer_weights
            else:
                weights = np.concatenate((weights, layer_weights))

    return weights


def vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    # TODO potentially slow...
    n_rows = np.floor(np.sqrt(vector.shape[0]))
    n_elements = vector.shape[0]
    while n_elements % n_rows != 0:
        n_rows = int(n_rows - 1)

    matrix = vector.reshape(n_rows, -1)
    return matrix


def plot_weights(E: torch.Tensor, D: torch.Tensor, out_path: Path, key: str):
    fig, axes = plt.subplots(2, figsize=(15, 15))
    sns.heatmap(E, ax=axes[0])
    axes[0].set_title(f"Encoder {np.prod(E.shape)} ({E.shape})")
    sns.heatmap(D, ax=axes[1])
    axes[1].set_title(f"Decoder {np.prod(D.shape)} ({D.shape})")

    plt_path = out_path / f"{key}_weights.png"
    plt.savefig(plt_path)
    plt.close()


def visualize_ae_activations(
    ae: torch.nn.Module,
    name: str = "ann",
    out_path: str = "",
    dl: torch.utils.data.DataLoader = None,
    data_type: str = "train",
    n_batches: int = 1,
) -> Tuple[dict[str, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
    # https://kozodoi.me/blog/20210527/extracting-features

    visualization = {}

    # TODO how to pass the name to the function instead of having two functions..
    def get_activation_name(name: str, ann: str, idx: int) -> callable:
        def hook_fn(m, i, o):
            visualization[f"{idx} {ann}: {name}"] = o

        return hook_fn

    for i, layer in enumerate(ae.encoder.encoder):
        layer.register_forward_hook(get_activation_name(layer, "FW Encoder", i))

    for i, layer in enumerate(ae.decoder.decoder):
        layer.register_forward_hook(get_activation_name(layer, "FW Decoder", i))

    all_batches = collections.defaultdict(list)
    stats = []
    for batch_idx, (data, _) in enumerate(dl):
        if batch_idx >= n_batches:
            break

        # TODO return more than one batch for comparison?
        print(name, batch_idx, data.shape)
        if torch.cuda.is_available():
            data = data.to("cuda")
        data = data.view(-1, ae.x_dim)
        visualization[INPUT_LAYER_NAME] = data
        # TODO hack
        recon_batch = ae(data)
        loss = ae.loss_function(recon_batch, data).item()
        # TODO add reconstruction output activations?
        # visualization["recon"] = recon_batch
        stats_data = {
            "measure_name": "loss",
            "measure_value": loss,
            "batch_idx": batch_idx,
            "name": name,
        }
        stats.append(stats_data)

        fig, axes = plt.subplots(len(visualization), 1, figsize=(25, 41))
        fig.add_gridspec(len(visualization), ncols=1, hspace=4)
        for i, data in enumerate(visualization.values()):
            vis_name = list(visualization.keys())[i]
            d = data.detach()
            sns.heatmap(d, ax=axes[i])
            axes[i].set_title(f"LAYER: {vis_name}")
            axes[i].set_ylabel("Exemplar Value")
            axes[i].set_xlabel("Dimension")
            out_file = out_path / f"{batch_idx}_{data_type}_{name}_ae_activations.png"
            all_batches[vis_name].append(d)
            D = d.numpy()
            stats_data = {
                "measure_name": f"{vis_name} std_0",
                "measure_value": np.sum(np.std(D, axis=0) == 0),
                "batch_idx": batch_idx,
                "name": name,
                "dimension": d.shape[1],
                "n_samples": d.shape[1],
            }

        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()

    for key, value in all_batches.items():
        values = value[0]
        for v in value[1:]:
            values = torch.concat((values, v))

        D = values.numpy()
        all_batches[key] = D.tolist()
        _stats_data = {
            "batch_idx": -1,
            "name": name,
            "measure_name": f"{key} std_0",
            "measure_value": np.sum(np.std(D, axis=0) == 0),
            "dimension": D.shape[1],
            "n_samples": D.shape[1],
        }
        stats.append(_stats_data)

    # Batch stats
    out_file = out_path / f"{name}_data.json"
    with open(out_file, "w", encoding="utf-8") as fd:
        json.dump(all_batches, fd)

    df = pd.DataFrame(stats)
    out_file = out_path / f"{name}_stats_data.jsonl"
    df.to_json(out_file, orient="records", lines=True)

    return visualization, all_batches, stats


def l2_norm_delta(M_0: torch.Tensor, M_1: torch.Tensor) -> float:
    l2_norm_diff = torch.linalg.vector_norm(M_0 - M_1, ord=2).data.item()
    return l2_norm_diff


def get_iteration_node_batch_from_ann_filename(file_name: str) -> tuple[int, int, int]:
    split_ = file_name.split("_")
    if len(split_) == 2:
        iteration, node, batch = split_[0][1:], "n-1", -1
    elif len(split_) < 5:
        iteration, node, batch = split_[1], split_[2], -1
    else:
        iteration, node, batch = split_[1], split_[2], split_[4]

    return int(iteration), int(node[1:]), int(batch)


def pair_plot(df: pd.DataFrame, out_path: Path):
    df_w = df[df["values"] == "weights"][["diff"]]
    df_w.columns = ["d_w"]
    df_w.reset_index(inplace=True, drop=True)
    value_names = df["values"].unique()
    for i, name in enumerate(value_names):
        if name == "weights":
            continue
        df_s = df[df["values"] == name][["diff"]]
        df_s.columns = [f"d_{i}"]
        df_s.reset_index(inplace=True, drop=True)
        df_w = pd.concat((df_w, df_s), axis=1)
        df_w.reset_index(inplace=True, drop=True)
    print(df_w.head())
    sns.pairplot(data=df_w)
    plt_path = out_path / "pair_plot.png"
    plt.savefig(plt_path)
    plt.close()


def coverage_histogram(df: pd.DataFrame, out_path: Path):
    _df = df[df["measure_value"] != 0]
    _df = _df[_df["measure_name"] != "loss"]
    _df = _df[_df["batch_idx"] == -1]
    _df["coverage ratio"] = 1.0 - _df["measure_value"] / _df["n_samples"]
    # Coverage count
    measure_names = _df["measure_name"].unique()
    ax = sns.histplot(
        _df,
        x="measure_value",
        hue="measure_name",
        multiple="stack",
        element="step",
        hue_order=measure_names,
    )
    legend = ax.get_legend()
    handles = legend.legend_handles
    assert len(handles) == len(measure_names)
    ax.legend(labels=measure_names, loc="upper left", bbox_to_anchor=(1, 0.5))
    plt_path = out_path / "hist_coverage_count_all.png"
    plt.savefig(plt_path)
    plt.close()
    # Coverage Ratio
    _df = _df[_df["coverage ratio"] < 1.0]
    measure_names = _df["measure_name"].unique()
    ax = sns.histplot(
        _df,
        x="coverage ratio",
        hue="measure_name",
        fill=True,
        multiple="dodge",
        element="step",
        hue_order=measure_names,
    )
    legend = ax.get_legend()
    handles = legend.legend_handles
    assert len(handles) == len(measure_names)
    ax.legend(labels=measure_names, loc="upper left", bbox_to_anchor=(1, 0.5))
    plt_path = out_path / "hist_coverage_ratio_all.png"
    plt.savefig(plt_path)
    plt.close()
    # Coverage Ratio vs Loss
    mean_values = (
        _df[_df["measure_name"] != "loss"].groupby(by="name")["coverage ratio"].mean()
    )
    losses = df[df["measure_name"] == "loss"].groupby(by="name")["measure_value"].sum()
    plt_df = pd.DataFrame()
    plt_df["mean coverage ratio"] = mean_values
    plt_df["sum loss"] = losses
    ax = sns.jointplot(data=plt_df, x="mean coverage ratio", y="sum loss")
    plt_path = out_path / "scatter_coverage_ratio_all.png"
    plt.savefig(plt_path)
    plt.close()


def main(
    network_folder: str = "analysis_tmp",
    ann_type: str = "AutoencoderBinaryClustering",
    batch_size: int = 2,
    dataset_name: str = "binary_clustering_10_100_1000",
    out_dir: str = "vis",
    data_prefix: str = "",
    max_anns: int = -1,
):
    networks = []
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)

    if data_prefix != "":
        logging.warning(f"Only picking the first folder with {data_prefix}.")
        network_paths = set()
        for root, dirnames, filenames in os.walk("."):
            if not data_prefix in root:
                continue
            for file in filenames:
                if file.endswith(".pt"):
                    network_paths.add(root)
                    network_file = Path(root) / file
                    networks.append(network_file)
    else:
        network_path = Path(network_folder)
        for file_path in network_path.glob("*.pt"):
            networks.append(file_path)

    print(network_paths)
    if max_anns > 0 and max_anns < len(networks):
        networks = networks[:max_anns]

    logging.info(f"Analyzing {len(networks)}")

    # Get networks
    anns = {}
    for network_path in networks:
        key = network_path.stem
        train_dl, test_dl, width, height, ae = load_data_and_autoencoder(
            batch_size=batch_size,
            ann_path=network_path,
            autoencoder=ann_type,
            dataset_name=dataset_name,
            data_loader_str="train",
        )
        anns[key] = ae

    weight_data = {}
    activation_data = {}
    all_batch_data = {}
    all_stats_data = []
    for key, ae in anns.items():
        # Weights
        E = get_weights(ae.encoder)
        E = vector_to_matrix(E)
        D = get_weights(ae.decoder)
        D = vector_to_matrix(D)
        weight_data[key] = (E, D)
        # TODO check if weights are too equal (as in not changing)
        plot_weights(E, D, out_path, key)
        # Activations
        activations, all_batches, all_stats = visualize_ae_activations(
            ae=ae, name=key, out_path=out_path, dl=train_dl, data_type="train"
        )
        activation_data[key] = activations
        all_batch_data[key] = all_batches
        all_stats_data.extend(all_stats)

    df = pd.DataFrame(all_stats_data)
    out_file = out_path / "all_stats.jsonl"
    df.to_json(out_file, orient="records", lines=True)
    coverage_histogram(df=df, out_path=out_path)

    # Inter diffs
    # TODO the combinations function in itertools makes sure that order does not matter for the combination? (That is why the heatmap is for a combinations since the order for the diff does not matter)
    # TODO order the combinations?
    combinations = itertools.combinations(weight_data.keys(), 2)
    diffs = []
    for combination in combinations:
        key_0, key_1 = combination
        iteration_0, node_0, batch = get_iteration_node_batch_from_ann_filename(key_0)
        iteration_1, node_1, batch = get_iteration_node_batch_from_ann_filename(key_1)
        ae_0 = anns[key_0]
        ae_1 = anns[key_1]
        delta = l2_norm_diff(ae_0, ae_1)
        diffs.append(
            {
                "values": "weights",
                "diff": delta,
                "key_0": key_0,
                "key_1": key_1,
                "iteration_0": iteration_0,
                "node_0": node_0,
                "iteration_1": iteration_1,
                "node_1": node_1,
                "batch": batch,
            }
        )

        for layer in activations.keys():
            if layer == INPUT_LAYER_NAME:
                continue
            a_0 = activation_data[key_0][layer][0]
            a_1 = activation_data[key_1][layer][0]
            delta = l2_norm_delta(a_0, a_1)
            diffs.append(
                {
                    "values": layer,
                    "diff": delta,
                    "key_0": key_0,
                    "key_1": key_1,
                    "iteration_0": iteration_0,
                    "node_0": node_0,
                    "iteration_1": iteration_1,
                    "node_1": node_1,
                    "batch": batch,
                }
            )

    df = pd.DataFrame(diffs)
    print(df)
    out_file = out_path / "interdiffs.jsonl"
    df.to_json(str(out_file), orient="records", lines=True)
    # Pair plot
    pair_plot(df, out_path)
    # Filter final best node
    max_iterations = df["iteration_0"].max()
    df = df[df["key_0"] != f"t{max_iterations}_ae"]
    df = df[df["key_1"] != f"t{max_iterations}_ae"]
    comparisons = df["values"].unique()
    for comparison in comparisons:
        for iteration_type in ("all", "only_0", "not_0"):
            _df = df[df["values"] == comparison]
            if iteration_type == "only_0":
                _df = _df[_df["iteration_0"] == 0]
                _df = _df[_df["iteration_1"] == 0]
            elif iteration_type == "not_0":
                _df = _df[_df["iteration_0"] > 0]
                _df = _df[_df["iteration_1"] > 0]
            # TODO hack. Duplicating data for doing lower triangle heatmap instead of sorting it
            _df_c = _df.copy()
            _df_c["key_0"] = _df["key_1"]
            _df_c["key_1"] = _df["key_0"]
            _df = pd.concat((_df, _df_c))
            data = _df.pivot_table(index="key_0", columns="key_1", values="diff")
            fig, axes = plt.subplots(1, figsize=(15, 15))
            plt_path = out_path / iteration_type
            os.makedirs(plt_path, exist_ok=True)
            plt_path = plt_path / f"{comparison}_{iteration_type}_heatmap.png"
            sns.heatmap(
                data,
                annot=True,
                ax=axes,
            )
            logging.info(f"Store {plt_path}")
            plt.savefig(plt_path)
            plt.close()

            # Loss diff

            # Performance diff

        # RQ Is there are correlation between diff and loss/performance? Would we expect the difference to be correlated with loss, since loss impacts the change of weights (yes within one network. But how about between them)
        # Networks start out random, and should then stabilize?
        # RQ Is the difference in network consistent over(inter) nodes and generations? Expect equal "divergence" of ANNs
        # RQ Is the in (intra) node divergence similar over time?
        # RQ Are there parts in the networks that are more volatile?


def parse_arguments(param: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments (`sys.argv`).
    """
    parser = argparse.ArgumentParser(description="Activation analysis of lipi-AE")
    parser.add_argument(
        "--data_prefix",
        type=str,
        help="Output data prefix. E.g. out_activation_analysis",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="binary_clustering_10_100_1000",
        help="Dataset name. E.g. binary_clustering_2_100_10",
    )
    parser.add_argument(
        "--max_anns",
        type=int,
        default=-1,
        help="Max number of ANNs to analyse. Useful for testing. E.g. 2",
    )
    args = parser.parse_args(param)
    return args


if __name__ == "__main__":
    param = parse_arguments(sys.argv[1:])
    main(
        data_prefix=param.data_prefix,
        dataset_name=param.dataset_name,
        max_anns=param.max_anns,
    )
