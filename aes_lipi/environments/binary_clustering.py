import logging
import os
import sys
import copy
from typing import Dict

import torch
from torchvision.utils import save_image
from torchmetrics.functional import retrieval_recall
from torchmetrics.functional.retrieval import retrieval_normalized_dcg

from aes_lipi.anns.ann import (
    Decoder,
    Encoder,
    Autoencoder,
    VariationalAutoencoder,
    get_activation_name,
)
from aes_lipi.datasets.data_loader import create_batches
from aes_lipi.environments.environment import parse_arguments


# From 7.3 Baldi A theory of local learning, the learning channel, and the optimality of backpropagation


class EncoderBinaryClustering(Encoder):
    def __init__(self, x_dim=100, z_dim=10, width=100, height=1) -> None:
        super(EncoderBinaryClustering, self).__init__(x_dim, z_dim, width, height)
        assert self.x_dim == self.height * self.width
        self.h_dim = 30
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.x_dim, self.h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.h_dim, self.z_dim),
            torch.nn.ReLU(True),
        )
        self.register_activation_hooks()

    def encode(self, x):
        h = self.encoder(x)
        self.stores.append(copy.deepcopy(self.store))
        return h

    @staticmethod
    def get_fixed_ann(**kwargs) -> torch.nn.Module:
        raise NotImplementedError(
            "Implement. Store an ANN and then load it. But not everytime"
        )


class EncoderBinaryClusteringSmall(EncoderBinaryClustering):
    def __init__(self, x_dim=100, z_dim=2, width=100, height=1) -> None:
        super(EncoderBinaryClusteringSmall, self).__init__(x_dim, z_dim, width, height)
        assert self.x_dim == self.height * self.width
        self.h_dim = 30
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.x_dim, self.h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.h_dim, self.z_dim),
            torch.nn.ReLU(True),
        )
        for i, layer in enumerate(self.encoder):
            layer.register_forward_hook(
                get_activation_name(layer, "FW Enc", i, self.store)
            )


class EncoderBinaryClusteringLarge(EncoderBinaryClustering):
    def __init__(self, x_dim=100, z_dim=10, width=100, height=1) -> None:
        super(EncoderBinaryClusteringLarge, self).__init__(x_dim, z_dim, width, height)
        assert self.x_dim == self.height * self.width
        self.h_dim = 30
        self.h_dim_2 = 60
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.x_dim, self.h_dim_2),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.h_dim_2, self.h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.h_dim, self.z_dim),
            torch.nn.ReLU(True),
        )
        for i, layer in enumerate(self.encoder):
            layer.register_forward_hook(
                get_activation_name(layer, "FW Enc", i, self.store)
            )


class VariationalEncoderBinaryClustering(Encoder):
    def __init__(self, x_dim=100, z_dim=10, width=100, height=1) -> None:
        super(VariationalEncoderBinaryClustering, self).__init__(
            x_dim, z_dim, width, height
        )
        self.h_dim = 30

        self.fc1 = torch.nn.Linear(self.x_dim, self.h_dim)
        self.fc21 = torch.nn.Linear(self.h_dim, self.z_dim)
        self.fc22 = torch.nn.Linear(self.h_dim, self.z_dim)

    def encode(self, x):
        h = torch.nn.functional.relu(self.fc1(x))
        mu = self.fc21(h)
        log_var = self.fc22(h)
        return mu, log_var


class DecoderBinaryClustering(Decoder):
    def __init__(self, x_dim=100, z_dim=10, width=100, height=1) -> None:
        super(DecoderBinaryClustering, self).__init__(x_dim, z_dim, width, height)
        assert self.x_dim == self.height * self.width
        self.h_dim = 30
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, self.h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.h_dim, self.x_dim),
            torch.nn.Sigmoid(),
        )
        self.register_activation_hooks()

    def decode(self, z):
        x_p = self.decoder(z)
        self.stores.append(copy.deepcopy(self.store))
        return x_p

    @staticmethod
    def get_fixed_ann(**kwargs) -> torch.nn.Module:
        raise NotImplementedError(
            "Implement. Store an ANN and then load it. But not everytime"
        )


class DecoderBinaryClusteringSmall(DecoderBinaryClustering):
    def __init__(self, x_dim=100, z_dim=2, width=100, height=1) -> None:
        super(DecoderBinaryClusteringSmall, self).__init__(x_dim, z_dim, width, height)
        assert self.x_dim == self.height * self.width
        self.h_dim = 30
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, self.h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.h_dim, self.x_dim),
            torch.nn.Sigmoid(),
        )
        for i, layer in enumerate(self.decoder):
            layer.register_forward_hook(
                get_activation_name(layer, "FW Enc", i, self.store)
            )


class DecoderBinaryClusteringLarge(DecoderBinaryClustering):
    def __init__(self, x_dim=100, z_dim=10, width=100, height=1) -> None:
        super(DecoderBinaryClusteringLarge, self).__init__(x_dim, z_dim, width, height)
        assert self.x_dim == self.height * self.width
        self.h_dim = 30
        self.h_dim_2 = 60
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, self.h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.h_dim, self.h_dim_2),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.h_dim_2, self.x_dim),
            torch.nn.Sigmoid(),
        )
        for i, layer in enumerate(self.decoder):
            layer.register_forward_hook(
                get_activation_name(layer, "FW Enc", i, self.store)
            )


class VariationalDecoderBinaryClustering(DecoderBinaryClustering):
    def __init__(self, x_dim=100, z_dim=10, width=100, height=1) -> None:
        super(VariationalDecoderBinaryClustering, self).__init__(
            x_dim, z_dim, width, height
        )
        self.h_dim = 30
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, self.h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.h_dim, self.x_dim),
            torch.nn.Sigmoid(),
        )


class AutoencoderBinaryClustering(Autoencoder):
    def __init__(self, encoder, decoder):
        super(AutoencoderBinaryClustering, self).__init__(encoder, decoder)
        self.L1_loss = torch.nn.L1Loss()

    def loss_function(self, x_p, x) -> torch.Tensor:
        L1 = self.L1_loss(x_p, x)
        self.encoder.loss = L1.data.item()
        self.decoder.loss = L1.data.item()
        return L1


class VariationalAutoencoderBinaryClustering(VariationalAutoencoder):
    pass


class DenoisingAutoencoderBinaryClustering(Autoencoder):
    def __init__(self, encoder, decoder):
        super(DenoisingAutoencoderBinaryClustering, self).__init__(encoder, decoder)
        self.L1_loss = torch.nn.L1Loss()

    def loss_function(self, x_p, x) -> torch.Tensor:
        assert torch.all(torch.where(torch.logical_and(0 <= x, x <= 1), True, False))
        assert torch.all(
            torch.where(torch.logical_and(0 <= x_p, x_p <= 1), True, False)
        )

        L1 = self.L1_loss(x_p, x)
        self.encoder.loss = L1.data.item()
        self.decoder.loss = L1.data.item()
        return L1


def train(ae, epoch, train_loader, optimizer):
    ae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            data = data.to("cuda")
        data = data.view(-1, ae.x_dim)
        # TODO hack
        if isinstance(ae, VariationalAutoencoder):
            recon_batch, mu, log_var = ae(data)
            loss = ae.loss_function(recon_batch, data, mu, log_var)
        else:
            recon_batch = ae(data)
            loss = ae.loss_function(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )
    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )
    if epoch % 50 == 0:
        x = recon_batch
        save_image(
            x.view(x.size(0), 1, ae.height, ae.width),
            f"./out_binary_problem/train_sample_{epoch}.png",
        )
        save_image(
            data.view(data.size(0), 1, ae.height, ae.width),
            f"./out_binary_problem/sample_{epoch}.png",
        )
        d = torch.abs(torch.sub(x, data))
        save_image(
            d.view(d.size(0), 1, ae.height, ae.width),
            f"./out_binary_problem/diff_{epoch}.png",
        )


def test(ae, epoch, test_loader):
    ae.eval()
    assert not ae.training
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            if torch.cuda.is_available():
                data = data.to("cuda")
            data = data.view(-1, ae.x_dim)

            if isinstance(ae, VariationalAutoencoder):
                recon_batch, mu, log_var = ae(data)
                loss = ae.loss_function(recon_batch, data, mu, log_var)
            else:
                recon_batch = ae(data)
                loss = ae.loss_function(recon_batch, data)

            # sum up batch loss
            test_loss += loss.item()
            retrieval_result = retrieval_measures(predicted=recon_batch, target=data)

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


def retrieval_measures(predicted, target) -> Dict[str, float]:
    k = 100
    retriveal_n100 = retrieval_recall(predicted, target, top_k=k)
    k = 20
    retriveal_n20 = retrieval_recall(predicted, target, top_k=k)
    k = 50
    dcg_n50 = retrieval_normalized_dcg(predicted, target, top_k=k)
    result = {
        "recall_n100": retriveal_n100.item(),
        "retrieval_n20": retriveal_n20.item(),
        "dcg_n50": dcg_n50.item(),
    }
    logging.info(f"Test k=[100, 20, 50] . {result}")
    return result


def main(method: str, dataset_name: str, epochs: int, save: bool = False):
    bs = 20

    train_loader, test_loader, width, height = create_batches(bs, dataset_name)
    # build model
    x_dim = width * height
    z_dim = 10
    if method == "Autoencoder":
        e = EncoderBinaryClustering(
            x_dim=x_dim, z_dim=z_dim, width=width, height=height
        )
        d = DecoderBinaryClustering(
            x_dim=x_dim, z_dim=z_dim, width=width, height=height
        )
        ae = AutoencoderBinaryClustering(e, d)
    elif method == "DenoisingAutoencoder":
        e = EncoderBinaryClustering(
            x_dim=x_dim, z_dim=z_dim, width=width, height=height
        )
        d = DecoderBinaryClustering(
            x_dim=x_dim, z_dim=z_dim, width=width, height=height
        )
        ae = DenoisingAutoencoderBinaryClustering(e, d)
    elif method == "VariationalAutoencoder":
        e = VariationalEncoderBinaryClustering(
            x_dim=x_dim, z_dim=z_dim, width=width, height=height
        )
        d = VariationalDecoderBinaryClustering(
            x_dim=x_dim, z_dim=z_dim, width=width, height=height
        )
        ae = VariationalAutoencoderBinaryClustering(e, d)
    else:
        raise Exception(f"Unknown method {method}")

    if torch.cuda.is_available():
        logging.info("Using CUDA")
        e.cuda()
        d.cuda()
        ae.cuda()

    optimizer = torch.optim.Adam(ae.parameters(), weight_decay=1e-5, lr=1e-2)

    for epoch in range(1, epochs):
        train(ae, epoch, train_loader, optimizer)
        test(ae, epoch, test_loader)

    with torch.no_grad():
        z = torch.randn(30, ae.z_dim)
        if torch.cuda.is_available():
            z = z.to("cuda")

        ae.decoder.decode(z)

    if save:
        data = {"encoder": e.state_dict(), "decoder": d.state_dict()}
        out_path = "ann.pt"
        torch.save(data, out_path)
        logging.info(f"Save to {out_path}")


if __name__ == "__main__":
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    os.makedirs("out_binary_problem", exist_ok=True)
    params = parse_arguments(sys.argv[1:])
    main(**vars(params))
