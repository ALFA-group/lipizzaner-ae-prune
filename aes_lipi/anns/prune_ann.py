import logging
import os
import sys
import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils import prune
from torch.utils.data import Subset

from aes_lipi.anns.ann import (
    get_activation_variance,
    reset_activation_store,
    get_activations,
)
from aes_lipi.datasets.data_loader import create_batches
from aes_lipi.environments.binary_clustering import (
    AutoencoderBinaryClustering,
    DecoderBinaryClustering,
    EncoderBinaryClustering,
)
from aes_lipi.environments.environment import parse_arguments


def main(method: str, dataset_name: str, epochs: int, save: bool = False):
    bs = 20

    train_loader, test_loader, width, height = create_batches(bs, dataset_name)
    # build model
    x_dim = width * height
    z_dim = 10
    assert method == "Autoencoder"
    e = EncoderBinaryClustering(x_dim=x_dim, z_dim=z_dim, width=width, height=height)
    d = DecoderBinaryClustering(x_dim=x_dim, z_dim=z_dim, width=width, height=height)
    ae = AutoencoderBinaryClustering(e, d)
    module = ae.encoder.encoder[0]
    for name, param in ae.encoder.named_parameters():
        if param.requires_grad:
            print(name)
    print("BEFORE", list(ae.encoder.named_parameters())[0:2])
    # Prune
    a = prune.random_unstructured(module, name="weight", amount=0.1)
    # TODO Use 1 - std as probability of being used in pruning mask
    print("PARAMS", list(module.named_parameters())[0:2])
    print("BUFFERS", list(module.named_buffers()))
    print("WEIGHT ATTRIBUTE", module.weight)
    w_mask = dict(module.named_buffers()).get("weight_mask")
    print("N Params", "Enc", get_n_params(ae.encoder), "Dec", get_n_params(ae.decoder))
    print("W", torch.count_nonzero(w_mask))
    print("W", w_mask.shape, np.sum(w_mask.detach().numpy() == 0))
    print("W", get_n_zero_parameters(module, "weight_mask"))
    prune.remove(module, "weight")
    print("W2", module.weight.shape, np.sum(module.weight.detach().numpy() == 0))
    print("AFTER REMOVE", list(module.named_parameters())[0:2])
    # Prune activations
    activation_values = torch.zeros((module.out_features, module.in_features))
    activations = {
        "0 FW Enc: Linear(in_features=1000, out_features=30, bias=True)": activation_values
    }
    prune_ann(ae.encoder.encoder, "activation", 0.1, activations=activations)


def get_n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_n_zero_parameters(module: torch.nn.Module, name: str) -> int:
    mask = dict(module.named_parameters()).get(name)
    if mask is not None:
        n_elements = mask.numel()
        non_zero_el = torch.count_nonzero(mask)
        zero_el = np.sum(mask.detach().cpu().numpy() == 0)
        assert non_zero_el + zero_el == n_elements
        logging.debug(
            f"{zero_el} {mask} for {name} in {module}. Buffers {dict(module.named_buffers())}"
        )
    else:
        logging.error(
            f"{mask} for {name} in {module}. Buffers {dict(module.named_buffers())}"
        )
        zero_el = 0
    return zero_el


class ActivationPruningMethod(prune.BasePruningMethod):
    """Prune based on activation variance"""

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # Get absolute values
        t_p = 1.0 - np.abs(t)
        min_t_p = np.min(t_p)
        ptp = np.max(t_p) - np.min(t_p)
        # Scale to [0, 1]
        if ptp > 0.0:
            t_p = (t_p - min_t_p) / ptp
        # TODO use threshold
        # TODO perturb ?
        rnd_values = np.random.random(t_p.shape)
        mask_p = t_p > rnd_values
        mask_p = torch.Tensor(mask_p.astype(int))
        return mask_p


class LexicasePruningMethod(ActivationPruningMethod):
    def compute_mask(self, t, default_mask):
        # Take selected nodes to prune tensor, invert it and return it as mask
        mask_p = torch.Tensor(np.invert(t).astype(int))  # TODO Probabilities?
        return mask_p


def activation_unstructured(
    module: torch.nn.Module,
    name: str,
    threshold: float,
    activations: Optional[torch.Tensor] = None,
) -> torch.nn.Module:
    try:
        ActivationPruningMethod.apply(module, name, importance_scores=activations)
    except Exception as e:
        print("Error", e)
        print(activations.shape)
        print(module.in_features, module.out_features)
        print(module.weight.shape)
        print(module.bias.shape)

    return module


def lexicase_unstructured(
    module: torch.nn.Module,
    name: str,
    threshold: float,
    activations: Optional[torch.Tensor] = None,
) -> torch.nn.Module:
    try:
        LexicasePruningMethod.apply(module, name, importance_scores=activations)
    except Exception as e:
        print("Error", e)
        print(activations.shape)
        print(module.in_features, module.out_features)
        print(module.weight.shape)
        print(module.bias.shape)

    return module


def prune_ann(
    ann: torch.nn.Module,
    prune_method: str,
    prune_amount: float,
    activations: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.nn.Module:
    n_pruned = 0
    for i, layer in enumerate(ann):
        activation_values = None
        for layer_activation in activations.keys():
            if layer_activation.startswith(f"{i} ") and layer_activation.endswith(
                f": {layer}"
            ):
                if prune_method == "lexicase":
                    activation_values = activations[layer_activation][
                        :, np.newaxis
                    ]  # Add required dimension for pruning
                else:
                    activation_values = activations[layer_activation]

        for name, param in list(layer.named_parameters()):
            if param.requires_grad:
                # TODO can we get this as an attribute instead of this hack
                property = name.split(".")[-1]
                if property in ("weight", "bias"):
                    module = layer
                    if prune_method == "random":
                        _ = prune.random_unstructured(module, property, prune_amount)
                        prune.remove(module, property)

                    elif prune_method == "activation":
                        if activation_values is None:
                            continue

                        if property == "weight":
                            activation_values_p = np.repeat(
                                activation_values, module.in_features, axis=1
                            )
                        else:
                            activation_values_p = activation_values[:, 0]

                        _ = activation_unstructured(
                            module,
                            property,
                            prune_amount,
                            activations=activation_values_p,
                        )
                        prune.remove(module, property)

                    elif prune_method == "lexicase":
                        if property == "weight":
                            activation_values_p = np.repeat(
                                activation_values, module.in_features, axis=1
                            )
                        else:
                            activation_values_p = activation_values[:, 0]
                        _ = lexicase_unstructured(
                            module,
                            property,
                            prune_amount,
                            activations=activation_values_p,
                        )
                        prune.remove(module, property)

                    n_pruned += get_n_zero_parameters(module, property)

    ann.n_pruned = n_pruned
    logging.info(f"{ann.n_pruned}")
    return ann


def lexicase_select_nodes(layer_activations, threshold):
    # Normalize activations and select nodes with values below threshold
    selected_list = []
    for layer in layer_activations:
        normalized_activations = (layer - np.min(layer)) / (
            np.max(layer) - np.min(layer)
        )
        selected_activations = normalized_activations < threshold
        selected_list.append(selected_activations)

    # Apply lexicase selection
    prev_selected = selected_list[0]
    select_count = np.sum(prev_selected)
    for i in range(0, len(selected_list)):
        if select_count <= 1:
            if i == 0:
                logging.info(
                    "All nodes fail first lexicase filter. No pruning this epoch."
                )
            break
        prev_selected = (
            prev_selected & selected_list[i]
        )  # Node only moves on if the node is selected in both this and previous filters
        select_count = np.sum(prev_selected)

    return prev_selected


def lexicase_select_activations(
    activation_dict: Dict[str, torch.Tensor],
    lexi_threshold: float = 0.1,
) -> Dict[str, torch.Tensor]:
    selection_threshold = lexi_threshold
    selected_activations_dict = {}
    for key in activation_dict.keys():
        layer_pruning_candidates = lexicase_select_nodes(
            activation_dict[key], selection_threshold
        )
        selected_activations_dict[key] = layer_pruning_candidates

    return selected_activations_dict


def prune_ae(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    prune_method: str,
    prune_amount: float,
    test_data: Optional[torch.utils.data.DataLoader] = None,
    lexi_threshold: float = 0.1,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    assert 0.0 <= prune_amount < 1.0
    if prune_method == "lexicase":
        num_lexicase_tests = 100
        subset_indicies = [
            random.randint(1, len(test_data.dataset) - 1)
            for _ in range(num_lexicase_tests)
        ]
        subset = Subset(test_data.dataset, subset_indicies)
        filter_cases = [item for item in subset]

        enc_lexi_store = []
        dec_lexi_store = []
        for x in filter_cases:
            ae = AutoencoderBinaryClustering(encoder, decoder)
            ae(x[0])
            enc_lexi_store.append(encoder.stores.pop())
            dec_lexi_store.append(decoder.stores.pop())

        enc_activations = lexicase_select_activations(
            get_activations(enc_lexi_store), lexi_threshold
        )
        dec_activations = lexicase_select_activations(
            get_activations(dec_lexi_store), lexi_threshold
        )
    else:
        enc_activations = get_activation_variance(encoder)
        dec_activations = get_activation_variance(decoder)

    try:
        encoder_p = prune_ann(
            encoder.encoder,
            prune_method,
            prune_amount,
            activations=enc_activations,
        )

        decoder_p = prune_ann(
            decoder.decoder,
            prune_method,
            prune_amount,
            activations=dec_activations,
        )
        logging.info(f"A P {encoder_p.n_pruned} {decoder_p.n_pruned}")
        encoder.n_pruned = str(encoder_p.n_pruned)
        decoder.n_pruned = str(decoder_p.n_pruned)
    except Exception as e:
        print(e)
        raise Exception(e)
    encoder.encoder = encoder_p
    decoder.decoder = decoder_p
    logging.info(f"AA P {encoder.encoder.n_pruned} {decoder.decoder.n_pruned}")
    logging.info(f"AA P {encoder.n_pruned} {decoder.n_pruned}")
    return encoder, decoder


def reset_ae_activations(
    encoder: torch.nn.Module, decoder: torch.nn.Module
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    reset_activation_store(encoder)
    reset_activation_store(decoder)
    return encoder, decoder


def fixed_schedule(probability: float, **kwargs) -> float:
    return probability


def increase_schedule(probability: float, **kwargs) -> float:
    """Increase probability and return it"""
    epoch = kwargs["epoch"]
    final_epoch = kwargs["final_epoch"]
    probability_per_epoch = probability / final_epoch
    current_probability = epoch * probability_per_epoch
    return current_probability


def decrease_schedule(probability: float, **kwargs) -> float:
    """Decrease probability and return it"""
    epoch = kwargs["epoch"]
    final_epoch = kwargs["final_epoch"]
    probability_per_epoch = probability / final_epoch
    current_probability = epoch * probability_per_epoch
    return 1.0 - current_probability


def final_n_schedule(probability: float, **kwargs) -> float:
    """Only return a probability in the last n% of epochs and return it"""
    epoch = kwargs["epoch"]
    final_epoch = kwargs["final_epoch"]
    n = 0.25  # TODO make it a parameter

    epoch_threshold = final_epoch - (n * final_epoch)

    if epoch < epoch_threshold:
        return 0
    else:
        return probability
    

def prune_after_schedule(probability: float, **kwargs) -> float:
    """Only prune during the final epoch"""
    epoch = kwargs["epoch"]
    final_epoch = kwargs["final_epoch"]

    epoch_threshold = final_epoch

    if epoch < epoch_threshold:
        return 0
    else:
        return probability


def exponential_schedule(probability: float, **kwargs) -> float:
    """Increase probability exponentially and return it"""
    epoch = kwargs["epoch"]
    final_epoch = kwargs["final_epoch"]
    steepness = 2  # TODO make it a parameter

    current_probability = probability * (1 - np.exp(-steepness * epoch / final_epoch))
    return current_probability


def population_schedule(probability: float, **kwargs) -> float:
    """Change probability based on population and return it"""
    # TODO make a guided implementation based on activation
    epoch = kwargs["epoch"]
    final_epoch = kwargs["final_epoch"]
    n_solutions = kwargs["n_solutions"]
    probability_per_epoch_and_solution = probability / (final_epoch * n_solutions)
    current_probability = epoch * probability_per_epoch_and_solution
    return current_probability


def get_prune_probability_from_schedule(schedule_name: str, **kwargs) -> Callable:
    match schedule_name:
        case "fixed":
            return fixed_schedule(**kwargs)
        case "increase":
            return increase_schedule(**kwargs)
        case "decrease":
            return decrease_schedule(**kwargs)
        case "population":
            return population_schedule(**kwargs)
        case "final_n":
            return final_n_schedule(**kwargs)
        case "prune_after":
            return prune_after_schedule(**kwargs)
        case "exponential":
            return exponential_schedule(**kwargs)


if __name__ == "__main__":
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    os.makedirs("out_prune_binary_problem", exist_ok=True)
    params = parse_arguments(sys.argv[1:])
    main(**vars(params))
