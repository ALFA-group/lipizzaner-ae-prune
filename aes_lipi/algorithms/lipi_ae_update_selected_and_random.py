import datetime
import itertools
import logging
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import DataLoader

from aes_lipi.algorithms.lipi_ae_base import (
    checkpoint_ae,
    device,
    evaluate_anns,
    loss_report_interval,
    replace,
    select,
    update_ann,
    get_neighbors,
    get_opponent,
    update_learning_rate,
)
from aes_lipi.anns.prune_ann import (
    get_prune_probability_from_schedule,
    prune_ae,
    reset_ae_activations,
)
from aes_lipi.utilities.utilities import (
    Node,
    measure_quality_scores,
    plot_node_losses,
    plot_train_data,
)


def evaluate_cells_all_vs_all(
    nodes: Dict[int, Node],
    epochs: int,
    visualize: str,
    stats: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_interval: int,
    score_keys: List[str],
    solution_concept: callable,
    training_data: DataLoader,
    **kwargs: Dict[str, Any],
) -> Dict[str, Node]:
    logging.info(f"Evaluate cell epoch outer-loop and node inner-loop all vs all")
    order = list(range(0, len(nodes)))
    kwargs["spatial"] = False
    # TODO clunky creating only 1 node
    node_0 = Node(
        learning_rate=nodes[0].learning_rate,
        index=0,
        encoders=[None],
        decoders=[None],
        Encoder=nodes[0].Encoder,
        Decoder=nodes[0].Decoder,
        Autoencoder=nodes[0].Autoencoder,
        kwargs=kwargs,
        training_data=training_data,
        optimizer=None,
    )
    encoders = [_.encoders[0] for _ in nodes.values()]
    decoders = [_.decoders[0] for _ in nodes.values()]
    fe_cnt = 0
    for t in range(epochs):
        pairs = itertools.product(order, order)

        for pair in pairs:
            idx_i, idx_j = pair
            encoder = encoders[idx_i]
            node_0.encoders[0] = encoder
            decoder = decoders[idx_j]
            node_0.decoders[0] = decoder
            assert len(node_0.encoders) == 1 and len(node_0.decoders) == 1
            # Cell evaluation
            node_0 = evaluate_cell(
                node_0,
                node_0.learning_rate,
                iteration=t,
                visualize=visualize,
                stats=stats,
                output_dir=output_dir,
                checkpoint_interval=checkpoint_interval,
                score_keys=score_keys,
                last_iteration=(epochs - 1),
                solution_concept=solution_concept,
                **kwargs,
            )
            stats[-1]["fe_cnt"] = fe_cnt
            fe_cnt += 1
        # Update learning rate

        node_0.learning_rate = update_learning_rate(node_0.learning_rate, **kwargs)
        np.random.shuffle(order)
        logging.debug(f"Shuffle cell evaluation order for epoch {t} to {order}")
        logging.info(f"Epoch {t} Min selection loss: {stats[-1]['min_selection_loss']}")

    return {0: node_0}


def evaluate_cells_epochs_nodes(
    nodes: Dict[int, Node],
    epochs: int,
    visualize: str,
    stats: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_interval: int,
    radius: int,
    score_keys: List[str],
    solution_concept: callable,
    **kwargs: Dict[str, Any],
):
    logging.info("Evaluate cells epoch outer-loop and node inner-loop")
    order = list(range(0, len(nodes)))
    fe_cnt = 0
    for t in range(epochs):
        assert not set(order).symmetric_difference(set(range(0, len(nodes)))), (
            f"{order}"
        )
        for idx in order:
            # Get neighborhood
            node = get_neighbors(nodes, nodes[idx], radius)

            # Cell evaluation
            node = evaluate_cell(
                node,
                node.learning_rate,
                iteration=t,
                visualize=visualize,
                stats=stats,
                output_dir=output_dir,
                checkpoint_interval=checkpoint_interval,
                score_keys=score_keys,
                last_iteration=(epochs - 1),
                solution_concept=solution_concept,
                **kwargs,
            )
            stats[-1]["fe_cnt"] = fe_cnt
            fe_cnt += 1
            # Update learning rate
            node.learning_rate = update_learning_rate(node.learning_rate, **kwargs)

        np.random.shuffle(order)
        logging.debug(f"Shuffle cell evaluation order for epoch {t} to {order}")


def evaluate_cell(
    node,
    learning_rate: float,
    iteration: int,
    visualize: str,
    stats: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_interval: int,
    score_keys: List[str],
    last_iteration: int,
    solution_concept: callable,
    **kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    logging.debug(
        f"Evaluate node {node.index} with learning rate {learning_rate} at iteration {iteration}"
    )
    # TODO messy use of kwargs. Should really make a class that can store instance variables...
    kwargs["output_dir"] = output_dir
    kwargs["node_idx"] = node.index
    kwargs["iteration"] = iteration

    training_data = node.training_data
    overwrite_checkpoint = kwargs.get("overwrite_checkpoint", True)
    stat = {
        "iteration": iteration,
        "learning_rate": learning_rate,
        "node_idx": node.index,
    }
    if (
        kwargs.get("spatial", True)
        and len(stats) > 0
        and stat["iteration"] == stats[-1]["iteration"]
    ):
        assert stat["node_idx"] != stats[-1]["node_idx"], (
            f"{stat} {stats[-1]} {len(stats)}\n{stats}"
        )

    # Get ANNs
    E, D = (node.encoders, node.decoders)
    # Get non evolving ANN
    # Get Random Batch
    # Evaluate
    losses, all_losses = evaluate_anns(
        E, D, training_data, node.Autoencoder, solution_concept
    )
    stat["min_selection_loss"] = np.min(losses)
    logging.debug(f"Min selection loss: {stat['min_selection_loss']} for {node.index}")
    if visualize == "all":
        plot_node_losses(all_losses, node.index, iteration, "select", output_dir)
    # Select
    E_p, D_p = select(node, losses)
    # TODO is is good to prune here?
    for i, (batch, _) in enumerate(training_data):
        logging.debug(f"Batch {i}")
        kwargs["batch_idx"] = i
        batch = batch.to(device)
        # Get random opponent
        d_p = get_opponent(D)
        for e in E_p:
            # Update ANN
            e, _, o = update_ann(
                e, d_p, batch, learning_rate, node.Autoencoder, node.optimizer, **kwargs
            )
            node.optimizer = o
        # Get random opponent
        e_p = get_opponent(E)
        for d in D_p:
            # Update ANN
            _, d, o = update_ann(
                e_p, d, batch, learning_rate, node.Autoencoder, node.optimizer, **kwargs
            )
            node.optimizer = o

    # Evaluate
    losses, all_losses = evaluate_anns(
        E_p, D_p, training_data, node.Autoencoder, solution_concept
    )
    if visualize == "all":
        plot_node_losses(all_losses, node.index, iteration, "replace", output_dir)

    if kwargs.get("prune_method", "None") != "None":
        prune_args = {
            "epoch": iteration,
            "final_epoch": last_iteration,
            "probability": kwargs["prune_probability"],
            "n_solutions": len(node.encoders),
        }
        prune_probability = get_prune_probability_from_schedule(
            kwargs["prune_schedule"], **prune_args
        )
        if prune_probability > np.random.random():
            for i in range(len(E_p)):
                E_p[i], D_p[i] = prune_ae(
                    E_p[i],
                    D_p[i],
                    kwargs["prune_method"],
                    kwargs.get("prune_amount", 0.0),
                    kwargs.get("test_data", None),
                )
                E_p[i], D_p[i] = reset_ae_activations(E_p[i], D_p[i])

        # Replace and set center
    node = replace(node, E_p, D_p, losses)
    stat["min_replacement_loss"] = np.min(losses)
    stat["timestamp"] = float(datetime.datetime.timestamp(datetime.datetime.now()))
    stat["n_pruned_encoder"] = int(node.encoders[0].encoder.n_pruned)
    stat["n_pruned_decoder"] = int(node.decoders[0].decoder.n_pruned)

    if kwargs.get("calculate_test_loss", False):
        test_data = kwargs.get("test_data", None)
        E, D = node.encoders[:1], node.decoders[:1]
        losses, all_losses = evaluate_anns(
            E, D, test_data, node.Autoencoder, solution_concept
        )
        stat["test_loss"] = np.min(losses)
        logging.info(f"Test loss: {stat['test_loss']}")

    fe_cnt = 0 if len(stats) == 0 else stats[-1]["fe_cnt"]
    stats.append(stat)
    logging.info(
        f"Epoch {iteration} Min selection loss: {stats[-1]['min_selection_loss']} fe_cnt: {fe_cnt}"
    )
    if iteration % loss_report_interval == 0 or iteration == last_iteration:
        ae = node.Autoencoder(node.encoders[0], node.decoders[0])
        logging.debug(f"{iteration} calculating quality scores on training_data")
        score_values = measure_quality_scores(score_keys, ae, training_data)
        stat.update(score_values)

        if visualize == "all":
            plot_train_data(node, batch, iteration, node.Autoencoder, output_dir)

        logging.debug(f"Min losses after training: {np.min(losses)}")

    if iteration % checkpoint_interval == 0:
        checkpoint_ae(
            output_dir=output_dir,
            encoder=node.encoders[0],
            decoder=node.decoders[0],
            t=iteration,
            node_idx=node.index,
            overwrite_checkpoint=overwrite_checkpoint,
        )

    return node
