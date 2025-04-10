import datetime
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from aes_lipi.algorithms.lipi_ae_base import checkpoint_ae, device, loss_report_interval
from aes_lipi.anns.prune_ann import (
    get_prune_probability_from_schedule,
    prune_ae,
    reset_ae_activations,
)
from aes_lipi.utilities.utilities import Node, measure_quality_scores


def evaluate_ann_canonical(
    nodes: Dict[int, Node],
    epochs: int,
    visualize: str,
    stats: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_interval: int,
    score_keys: List[str],
    **kwargs: Dict[str, Any],
) -> Dict[str, Node]:
    logging.info(f"Evaluate ann canonical")
    assert len(nodes) == 1
    training_data = nodes[0].training_data
    checkpoint_every_update = kwargs.get("checkpoint_every_update", False)
    # TODO is this subpoptimal use of the optimizer?
    ae = nodes[0].Autoencoder(nodes[0].encoders[0], nodes[0].decoders[0])
    ae = ae.to(device)
    learning_rate = nodes[0].learning_rate
    optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(epochs / 4)), gamma=0.1
    )

    fe_cnt = 0
    ae.train()
    for t in tqdm(range(epochs), position=2, desc="Epochs"):
        stat = {
            "iteration": t,
            "learning_rate": learning_rate,
            "node_idx": nodes[0].index,
            "fe_cnt": fe_cnt,
        }
        losses = []
        for i, (batch, _) in enumerate(training_data):
            # if checkpoint_every_update:
            #     checkpoint_ae(
            #         output_dir=output_dir,
            #         encoder=ae.encoder,
            #         decoder=ae.decoder,
            #         node_idx=nodes[0].index,
            #         t=t,
            #         overwrite_checkpoint=kwargs.get("overwrite_checkpoint", True),
            #         batch_idx=i,
            #     )

            batch = batch.to(device)
            optimizer.zero_grad()
            loss = ae.compute_loss_together(batch.view(-1, ae.x_dim))
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().cpu().numpy())

        if kwargs.get("prune_method", "None") != "None":
            prune_args = {
                "epoch": t,
                "final_epoch": epochs,
                "probability": kwargs["prune_probability"],
                "n_solutions": 1,
            }
            prune_probability = get_prune_probability_from_schedule(
                kwargs["prune_schedule"], **prune_args
            )
            if prune_probability > np.random.random():
                e_p, d_p = prune_ae(
                    ae.encoder,
                    ae.decoder,
                    kwargs["prune_method"],
                    kwargs.get("prune_amount", 0.0),
                    kwargs.get("test_data", None),
                )
                e_p, d_p = reset_ae_activations(e_p, d_p)
                ae.encoder = e_p
                ae.decoder = d_p

        fe_cnt += 1
        stat["min_selection_loss"] = float(np.min(losses))
        stat["min_replacement_loss"] = float(np.min(losses))
        stat["timestamp"] = float(datetime.datetime.timestamp(datetime.datetime.now()))
        stat["n_pruned_encoder"] = int(nodes[0].encoders[0].n_pruned)
        stat["n_pruned_decoder"] = int(nodes[0].decoders[0].n_pruned)

        if t % loss_report_interval == 0 or t == (epochs - 1):
            logging.info(f"{t} at loss report interval. {nodes[0].Encoder}")
            logging.info(f"{t} calculating quality scores on training_data")
            score_values = measure_quality_scores(score_keys, ae, training_data)
            stat.update(score_values)

        if t % checkpoint_interval == 0:
            checkpoint_ae(
                output_dir=output_dir,
                encoder=nodes[0].encoders[0],
                decoder=nodes[0].decoders[0],
                t=t,
                node_idx=nodes[0].index,
                overwrite_checkpoint=kwargs.get("overwrite_checkpoint", True),
            )

        # Update learning rate
        if not kwargs.get("constant_learning_rate", False):
            try:
                scheduler.step()
            except ZeroDivisionError as e:
                logging.error(e)
        stats.append(stat)
        learning_rate = scheduler.get_lr()[0]
        logging.info(f"Epoch {t} Min selection loss: {stats[-1]['min_selection_loss']}")

    return nodes
