# TODO make ABC as well?
import copy
import logging
import os
from typing import Dict

import numpy as np
import torch


def get_activation_name(
    name: str, ann: str, idx: int, store: Dict[str, list]
) -> callable:
    def hook_fn(m, i, o):
        key = f"{idx} {ann}: {name}"
        store.update({key: o.detach()})

    return hook_fn


def reset_activation_store(ann: torch.nn.Module) -> None:
    ann.stores = []
    ann.activations = {}


def get_activations(lexi_store):
    all_values = {}
    for key in lexi_store[0].keys():
        temp_values = np.zeros((len(lexi_store), *lexi_store[0][key].shape))
        for j, values in enumerate(lexi_store):
            D = values[key].numpy()
            temp_values[j] = D
        all_values[key] = temp_values

    return all_values


def get_activation_variance(ann: torch.nn.Module) -> Dict[str, torch.Tensor]:
    all_stds = {}
    for key in ann.store.keys():
        all_values = np.zeros((len(ann.stores), *ann.store[key].shape))
        for i, values in enumerate(ann.stores):
            D = values[key].numpy()
            all_values[i] = D

        std_v = np.sum(np.std(all_values, axis=0), axis=0).reshape(-1, 1)
        all_stds[key] = std_v

    return all_stds


class Encoder(torch.nn.Module):
    def __init__(self, x_dim, z_dim, width, height):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        self.height = height
        self.width = width
        assert self.x_dim == self.height * self.width
        self.z_dim = z_dim
        self.loss = None

        self.store = {}
        self.stores = []
        self.n_pruned = 0

    def register_activation_hooks(self):
        self.store = {}
        for i, layer in enumerate(self.encoder):
            self.store[f"{i} FW Enc: {layer}"] = torch.Tensor()
            layer.register_forward_hook(
                get_activation_name(layer, "FW Enc", i, self.store)
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def clear_stores(self):
        self.store = {}
        self.stores = []

    def clone(self) -> torch.nn.Module:
        my_clone = type(self)(self.x_dim, self.z_dim, self.width, self.height)
        my_clone.load_state_dict(self.state_dict())
        my_clone.register_activation_hooks()
        my_clone.n_pruned = self.n_pruned        
        return my_clone


class Decoder(torch.nn.Module):
    def __init__(self, x_dim, z_dim, width, height):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.height = height
        self.width = width
        assert self.x_dim == self.height * self.width
        self.z_dim = z_dim
        self.loss = None

        self.store = {}
        self.stores = []
        self.n_pruned = 0

    def register_activation_hooks(self):
        self.store = {}
        for i, layer in enumerate(self.decoder):
            self.store[f"{i} FW Dec: {layer}"] = torch.Tensor()
            layer.register_forward_hook(
                get_activation_name(layer, "FW Dec", i, self.store)
            )

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def clear_stores(self):
        self.store = {}
        self.stores = []

    def clone(self) -> torch.nn.Module:
        my_clone = type(self)(self.x_dim, self.z_dim, self.width, self.height)
        my_clone.load_state_dict(self.state_dict())
        my_clone.register_activation_hooks()
        my_clone.n_pruned = self.n_pruned
        return my_clone


class Autoencoder(torch.nn.Module):
    # TODO better names, might not always be encoder decoder?
    # TODO what are the arguments?
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(Autoencoder, self).__init__()
        self.Decoder = None
        self.Encoder = None
        self.encoder = encoder
        self.decoder = decoder
        self.x_dim = encoder.x_dim
        self.z_dim = decoder.z_dim
        self.height = encoder.height
        self.width = encoder.width

    def forward(self, x):
        z = self.encoder.encode(x)
        x_p = self.decoder.decode(z)
        return x_p

    def compute_loss_together(self, x: torch.Tensor) -> torch.Tensor:
        x_p = self.forward(x)
        loss = self.loss_function(x_p, x)

        return loss

    def loss_function(self, x_p, x, *args) -> torch.Tensor:
        pass


class DenoisingAutoencoder(Autoencoder):
    def __init__(self, encoder, decoder):
        super(DenoisingAutoencoder, self).__init__(encoder, decoder)

        self.noise_level = 0.4

    def add_noise(self, x) -> torch.Tensor:
        # TODO more than random noise
        noise = torch.randn(x.size()) * self.noise_level
        _x = x + noise
        return _x

    # https://github.com/shentianxiao/text-autoencoders/blob/master/model.py
    def forward(self, x):
        _x = self.add_noise(x) if self.training else x
        z = self.encoder.encode(_x)
        x_p = self.decoder.decode(z)
        return x_p


class VariationalAutoencoder(Autoencoder):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__(encoder, decoder)

    def forward(self, x):
        mu, log_var = self.encoder.encode(x)
        z = reparameterize(mu, log_var)
        x_p = self.decoder.decode(z)
        return x_p, mu, log_var

    def compute_loss_together(self, x: torch.Tensor) -> torch.Tensor:
        x_p, mean, logvar = self.forward(x)
        loss = self.loss_function(x_p, x, mean, logvar)

        return loss

    def loss_function(self, x_p, x, mu, log_var):
        x_p = torch.clamp(x_p, max=1.0)
        inf_mask = torch.isinf(x_p)
        if inf_mask.any():
            logging.warning("Inf in reconstruction")

        nan_mask = torch.isnan(x_p)
        if nan_mask.any():
            logging.warning("Nan in reconstruction")
            x_p = torch.nan_to_num(x_p, nan=1.0)

        x_p = torch.clamp(x_p, min=0.0, max=1.0)
        BCE = torch.nn.functional.binary_cross_entropy(x_p, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = BCE + KLD
        nan_mask = torch.isnan(loss)
        if nan_mask.any():
            logging.warning("Nan in loss")
            loss = torch.nan_to_num(loss, nan=10000000)

        return loss


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = eps.mul(std).add_(mu)
    return z


def add_noise(x, noise_level) -> torch.Tensor:
    # TODO more than random noise
    noise = torch.randn(x.size()) * noise_level
    _x = x + noise
    return _x


if __name__ == "__main__":
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
