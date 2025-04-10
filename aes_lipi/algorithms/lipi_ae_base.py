import logging
import os
import copy
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from aes_lipi.utilities.utilities import (
    Node,
    get_autoencoder,
    get_dataset_sample,
    plot_ae_data,
    plot_node_losses,
)


dtype = torch.float32
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

loss_report_interval = 100

rng = np.random.default_rng()


def checkpoint_ae(
    output_dir: str,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    node_idx: int,
    t: int,
    overwrite_checkpoint: bool,
    batch_idx: int = -1,
) -> None:
    out_path = os.path.join(output_dir, "checkpoints", f"n{node_idx}_ae.pt")
    if not overwrite_checkpoint:
        file_name = f"t_{t}_n{node_idx}_ae.pt"
        if batch_idx > -1:
            file_name = file_name.replace("_ae.pt", f"_b_{batch_idx}_ae.pt")

        out_path = os.path.join(output_dir, "checkpoints", file_name)

    data = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
    }
    torch.save(data, out_path)
    logging.debug(f"Checkpoint ANNs for {node_idx} in {out_path}")


def get_best_nodes(
    nodes: List[Node],
    test_data: DataLoader,
    visualize: str,
    iteration: int,
    output_dir: str,
) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
    encoders = [node.encoders[0] for node in nodes]
    decoders = [node.decoders[0] for node in nodes]
    all_losses = np.zeros((len(encoders), 1, 1))
    for data, _ in test_data:
        data = data.to(device)
        for i, (e, d) in enumerate(zip(encoders, decoders)):
            # TODO assumes all nodes have the same autoencoder class
            ae = nodes[0].Autoencoder(e, d)
            loss = ae.compute_loss_together(data.view(-1, ae.x_dim))
            all_losses[i, 0, 0] = all_losses[i, 0, 0] + loss.data.item()

    e_sorted_idx = np.argsort(all_losses)
    idx = e_sorted_idx[0, 0, 0]
    best_encoder = encoders[idx]
    best_decoder = decoders[idx]
    if visualize == "all":
        plot_node_losses(all_losses, idx, iteration, "best", output_dir)
    if visualize != "none":
        plot_ae_data(ae, data, "test", output_dir)

    logging.info(f"Get best ANNs with losses {all_losses}")
    return best_encoder, best_decoder, idx


def calculate_losses(
    encoders: List[torch.nn.Module],
    decoders: List[torch.nn.Module],
    data: torch.Tensor,
    Autoencoder: torch.nn.Module,
) -> np.ndarray:
    # Sum loss, encoder loss, decoder loss
    all_losses = np.zeros((len(encoders), len(decoders), 3))
    for i, encoder in enumerate(encoders):
        for j, decoder in enumerate(decoders):
            ae = Autoencoder(encoder, decoder)
            ae = ae.to(device)
            # TODO here set up for activation logging

            loss = ae.compute_loss_together(data)
            all_losses[i, j, 0] = loss
            all_losses[i, j, 1] = ae.encoder.loss
            all_losses[i, j, 2] = ae.decoder.loss

    return all_losses


def worst_case(all_losses: np.array) -> np.array:
    return np.max(all_losses[:, :, 0], axis=1)


def best_case(all_losses: np.array) -> np.array:
    return np.min(all_losses[:, :, 0], axis=1)


def mean_expected_utility(all_losses: np.array) -> np.array:
    return np.mean(all_losses[:, :, 0], axis=1)


def evaluate_anns(
    encoders: List[torch.nn.Module],
    decoders: List[torch.nn.Module],
    input_data: DataLoader,
    Autoencoder: torch.nn.Module,
    solution_concept: callable,
) -> Tuple[np.ndarray, np.ndarray]:
    # Evaluates on one batch
    # TODO assumes all encoders the same
    data = next(iter(input_data))[0].view(-1, encoders[0].x_dim)
    data = data.to(device)
    all_losses = calculate_losses(encoders, decoders, data, Autoencoder)
    # Get the worst case
    # TODO need both encoder and decoder loss
    losses = solution_concept(all_losses)
    logging.debug(
        f"Evaluated on one batch with {solution_concept} losses {all_losses.shape} {losses.shape} max losses: {losses} (min loss for all {np.min(all_losses[:, :, 0], axis=1)})"
    )
    return losses, all_losses


def update_ann(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    batch: torch.Tensor,
    learning_rate: float,
    Autoencoder: torch.nn.Module,
    optimizer: Any = None,
    **kwargs,
) -> torch.nn.Module:
    logging.debug(f"BEGIN Update ANN {encoder.__class__.__name__}")
    if kwargs.get("checkpoint_every_update", False):
        checkpoint_ae(
            output_dir=kwargs["output_dir"],
            encoder=encoder,
            decoder=decoder,
            t=kwargs.get("iteration", -1),
            overwrite_checkpoint=kwargs.get("overwrite_checkpoint", True),
            node_idx=kwargs.get("node_idx", -1),
            batch_idx=kwargs.get("batch_idx", -1),
        )

    ae = Autoencoder(encoder, decoder)
    ae = ae.to(device)
    ae.train()
    # TODO is optimizer used correctly. E.g. should we save the optimizer state, even though the encoder and decoder can be different
    optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    optimizer.zero_grad()
    # TODO MNIST hardcoding
    # TODO here set up for activation logging ??

    loss = ae.compute_loss_together(batch.view(-1, ae.x_dim))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ae.parameters(), sys.float_info.max / 1.0e10)
    optimizer.step()

    return encoder, decoder, optimizer


def select(
    node: Node, losses: np.ndarray
) -> Tuple[List[torch.nn.Module], List[torch.nn.Module]]:
    encoders = []
    decoders = []
    tournament_size = 2
    sub_population_size = 1
    for _ in range(sub_population_size):
        idxs = rng.integers(0, losses.shape[0], size=(tournament_size, 1))
        # TODO tournament size other than 2
        assert idxs.shape == (2, 1)
        # Check loss for the selected individuals
        if losses[idxs[0, 0]] < losses[idxs[1, 0]]:
            idx = idxs[0, 0]
        else:
            idx = idxs[1, 0]

        selected_encoder = node.encoders[idx]
        selected_decoder = node.decoders[idx]

        encoders.append(selected_encoder)
        decoders.append(selected_decoder)

    logging.debug(f"Selected {len(encoders)} from node {node.index}.")
    return encoders, decoders


def replace(
    node: Node,
    encoders: List[torch.nn.Module],
    decoders: List[torch.nn.Module],
    losses: np.ndarray,
) -> Node:
    e_sorted_idx = np.argsort(losses)
    node.encoders = [encoders[_] for _ in e_sorted_idx]
    d_sorted_idx = np.argsort(losses)
    node.decoders = [decoders[_] for _ in d_sorted_idx]
    logging.debug(
        f"Replace anns with ({(e_sorted_idx, d_sorted_idx)}) in node {node.index}"
    )
    return node


def get_opponent(anns: List[torch.nn.Module]) -> torch.nn.Module:
    opponent = rng.choice(anns)
    logging.debug(f"Opponent {opponent.__class__.__name__}")
    return opponent


def update_learning_rate(learning_rate: float, step_std=0.00001, **kwargs) -> float:
    # TODO todo use pytorch scheduler. Seems like could be more SE to get it to work with optimizers
    # TODO better setting of step size
    MIN_LR = 0.00001 if kwargs.get("sgd_optimizer", False) else 0.000000001
    MAX_LR = 10
    delta = rng.normal(0, step_std)
    learning_rate = max(MIN_LR, learning_rate + delta)
    logging.debug(f"lr: {learning_rate}, delta: {delta}")
    # TODO lower learning rate to avoid errors
    learning_rate = min(MAX_LR, learning_rate)

    logging.debug(
        f"Update learning rate to {learning_rate} original {kwargs.get('original_learning_rate')}"
    )
    return learning_rate


def get_neighbors(nodes: Dict[int, Node], node: Node, radius: int) -> Node:
    encoders = [node.encoders[0]]
    decoders = [node.decoders[0]]
    idx = [node.index]
    for i in range(1, radius + 1):
        # Right
        r_idx = (node.index + i) % len(nodes)
        neighbor = nodes[r_idx].encoders[0]
        encoders.append(neighbor.clone())
        neighbor = nodes[r_idx].decoders[0]
        decoders.append(neighbor.clone())
        l_idx = (node.index - i) % len(nodes)
        # Left
        neighbor = nodes[l_idx].encoders[0]
        encoders.append(neighbor.clone())
        neighbor = nodes[l_idx].decoders[0]
        decoders.append(neighbor.clone())
        idx.extend((r_idx, l_idx))

    node.encoders = encoders
    node.decoders = decoders
    assert len(node.decoders) == len(node.encoders) == ((2 * radius) + 1)
    logging.debug(
        f"Get {radius * 2} neighbors ({idx}) from {len(nodes)} nodes for node {node.index}"
    )

    return node


def initialize_nodes(
    learning_rate: float,
    population_size: int,
    environment: str,
    ann_path: str,
    training_data: DataLoader,
    width: int,
    height: int,
    dataset_fraction: float = 0.0,
    identical_initial_ann: bool = False,
) -> Dict[int, Node]:
    nodes = {}
    # TODO ugly. Maybe make node a normal class instead of a data class. Also look at Lipi SE
    Autoencoder, Encoder, Decoder, kwargs = get_autoencoder(
        environment, training_data, width, height
    )
    dataset = get_dataset_sample(training_data, dataset_fraction)
    logging.info(f"Node kwargs: {kwargs}")
    for i in range(population_size):
        encoder = Encoder(**kwargs)
        decoder = Decoder(**kwargs)
        if identical_initial_ann and i > 0:
            encoder = nodes[0].encoders[0].clone()
            decoder = nodes[0].decoders[0].clone()
            logging.info(f"{identical_initial_ann} cloning ANNs from 0 to {i}")

        node = Node(
            learning_rate=learning_rate,
            index=i,
            encoders=[encoder],
            decoders=[decoder],
            # TODO remove Encoder, Decoder fields and take them from Autoencoder instead?
            Encoder=Encoder,
            Decoder=Decoder,
            Autoencoder=Autoencoder,
            kwargs=kwargs,
            training_data=dataset,
            optimizer=None,
        )
        node.encoders[0] = node.encoders[0].to(device)
        node.decoders[0] = node.decoders[0].to(device)
        nodes[i] = node
        if ann_path != "":
            file_path = ann_path.replace("IDX", str(i))
            data = torch.load(file_path)
            nodes[i].encoders[0].load_state_dict(data["encoder"])
            nodes[i].decoders[0].load_state_dict(data["decoder"])
            logging.debug(f"Loading ANNs from {file_path}")

    logging.debug(
        f"Initialized {population_size} nodes with learning rate {learning_rate} from {environment}"
    )
    return nodes
