import logging
import os
import sys

import torch
from torch.distributions import binomial, uniform
from torch.utils.data import Dataset
from torchvision.utils import save_image

from aes_lipi.anns.ann import (
    Decoder,
    DenoisingAutoencoder,
    Encoder,
    Autoencoder,
    VariationalAutoencoder,
)
from aes_lipi.datasets.data_loader import create_batches
from aes_lipi.environments.binary_clustering import (
    AutoencoderBinaryClustering,
    DecoderBinaryClustering,
    DenoisingAutoencoderBinaryClustering,
    EncoderBinaryClustering,
    VariationalAutoencoderBinaryClustering,
    VariationalDecoderBinaryClustering,
    VariationalEncoderBinaryClustering,
)
from aes_lipi.environments.environment import parse_arguments

# From 7.3 Baldi A theory of local learning, the learning channel, and the optimality of backpropagation


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

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


def main(method: str, dataset_name: str, epochs: int = 2):
    bs = 2000
    assert dataset_name == "msd"
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


if __name__ == "__main__":
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    os.makedirs("out_retrieval", exist_ok=True)
    params = parse_arguments(sys.argv[1:])
    main(**vars(params))
