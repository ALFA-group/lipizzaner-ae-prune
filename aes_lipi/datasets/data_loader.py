import argparse
import itertools
import logging
import os
import sys
from typing import List, Optional, Tuple

import requests
import torch
from torch.distributions import binomial, uniform
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd

from aes_lipi.datasets.retrieval_msd import get_unique_sid, load_train_data


device = "cpu"
dtype = torch.float32


def create_binary_cluster_data(
    n_clusters: int = 10,
    n_dim: int = 100,
    noise_probability: float = 0.05,
    p: float = 0.5,
    cluster_centroids: Optional[torch.Tensor] = None,
    distribution: Optional[binomial.Binomial] = None,
    n_examples_per_cluster: int = 100,
) -> str:
    logging.info(
        f"Create binary cluster data with n_dim: {n_dim} and n_clusters: {n_clusters}"
    )
    if distribution is None:
        distribution = binomial.Binomial(2, probs=p)
    if cluster_centroids is None:
        cluster_centroids = distribution.sample((n_clusters, n_dim))
    uniform_d = uniform.Uniform(0, 1)
    mask = (
        uniform_d.sample((n_clusters, n_examples_per_cluster, n_dim))
        < noise_probability
    )
    _n_samples = int(n_examples_per_cluster / n_clusters)
    assert n_examples_per_cluster % n_clusters == 0, (
        f"n_examples_per_cluster must be dividable by n_clusters"
    )
    samples = cluster_centroids.repeat(n_clusters, _n_samples, 1)
    print(
        f"Cluster centroids: {cluster_centroids.shape}, mask: {mask.shape}, sample: {samples.shape}"
    )
    train_d = torch.logical_xor(samples, mask)
    mask = uniform_d.sample((n_examples_per_cluster, n_dim)) < noise_probability
    test_d = torch.logical_xor(samples, mask)
    mask = uniform_d.sample((n_examples_per_cluster, n_dim)) < noise_probability
    validation_d = torch.logical_xor(samples, mask)

    folder_name = "binary_problem_data"
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"binary_clustering_{n_clusters}_{n_examples_per_cluster}_{n_dim}.pt"

    datasets = {"train": train_d, "test": test_d, "validation": validation_d}
    for key, value in datasets.items():
        out_file = os.path.join(folder_name, f"{key}_{file_name}")
        torch.save(value, out_file)
        logging.info(f"Save {out_file}")

        out_file = os.path.join(folder_name, f"{key}_T_{file_name}")
        torch.save(value.view(-1, n_dim), out_file)
        logging.info(f"Save {out_file}")

    return file_name


class BinaryClusterDataset(Dataset):
    def __init__(self, data_file) -> None:
        super(BinaryClusterDataset).__init__()
        self.data = torch.load(data_file, weights_only=False).to(torch.float32)
        self.data_file = data_file
        logging.info(f"Loading {self.data.shape} from {data_file}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx, :]
        return sample, idx


class MSDDataset(Dataset):
    # TODO use spars.csr_matrix format?
    def __init__(self, data_file) -> None:
        super(MSDDataset).__init__()
        data_folder = os.path.dirname(data_file)
        n_items = len(get_unique_sid(pro_dir=data_folder))
        self.data = load_train_data(data_file, n_items=n_items, sparse_m=False)
        self.data_file = data_file
        logging.info(f"Loading {self.data.shape} from {data_file}")

    def __len__(self) -> int:
        return self.data.shape[0]

    def get_n_dim(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx, :]
        return sample, idx


class PendigitsDataset(Dataset):
    """UCI Pen-Based Recognition of Handwritten Digits Dataset.
    
    16 features representing (x,y) coordinates of 8 points along pen trajectory.
    """
    def __init__(self, split='train', data_path='') -> None:
        super(PendigitsDataset).__init__()
        self.split = split
        
        # Construct file path
        if data_path:
            base_path = os.path.join(data_path, 'data', 'pendigits')
        else:
            base_path = 'data/pendigits'
        
        file_name = f'{split}.csv' if split in ['train', 'test'] else 'train.csv'
        file_path = os.path.join(base_path, file_name)
        
        # Load and prepare data
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, header=None)
            # Last column is the label, first 16 are features
            self.data = torch.tensor(data.iloc[:, :16].values, dtype=torch.float32)
            self.labels = torch.tensor(data.iloc[:, 16].values, dtype=torch.long)
        else:
            # Try downloading if file doesn't exist
            self._download_data(base_path)
            data = pd.read_csv(file_path, header=None)
            self.data = torch.tensor(data.iloc[:, :16].values, dtype=torch.float32)
            self.labels = torch.tensor(data.iloc[:, 16].values, dtype=torch.long)
        
        # Feature-wise min-max normalization to [0, 1]
        col_min = self.data.min(dim=0).values
        col_max = self.data.max(dim=0).values
        # Avoid division by zero for constant columns
        denom = torch.where((col_max - col_min) == 0, torch.ones_like(col_max), (col_max - col_min))
        self.data = (self.data - col_min) / denom
        
        logging.info(f"Loading pendigits {split}: {self.data.shape} from {file_path}")
    
    def _download_data(self, base_path: str) -> None:
        """Download pendigits dataset from UCI repository."""
        os.makedirs(base_path, exist_ok=True)
        
        urls = {
            'train': 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra',
            'test': 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes'
        }
        
        for split, url in urls.items():
            file_path = os.path.join(base_path, f'{split}.csv')
            logging.info(f"Downloading {split} data from {url}")
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Saved to {file_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx, :]
        return sample, idx


class ShuttleDataset(Dataset):
    """UCI Statlog (Shuttle) Dataset.
    
    9 features representing sensor readings from shuttle flights.
    """
    def __init__(self, split='train', data_path='') -> None:
        super(ShuttleDataset).__init__()
        self.split = split
        
        # Construct file path
        if data_path:
            base_path = os.path.join(data_path, 'data', 'shuttle')
        else:
            base_path = 'data/shuttle'
        
        file_name = f'{split}.csv' if split in ['train', 'test'] else 'train.csv'
        file_path = os.path.join(base_path, file_name)
        
        # Load and prepare data
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, header=None, sep=' ', skipinitialspace=True)
            # Last column is the label, first 9 are features
            self.data = torch.tensor(data.iloc[:, :9].values, dtype=torch.float32)
            self.labels = torch.tensor(data.iloc[:, 9].values, dtype=torch.long)
        else:
            # Try downloading if file doesn't exist
            self._download_data(base_path)
            data = pd.read_csv(file_path, header=None, sep=' ', skipinitialspace=True)
            self.data = torch.tensor(data.iloc[:, :9].values, dtype=torch.float32)
            self.labels = torch.tensor(data.iloc[:, 9].values, dtype=torch.long)
        
        # Feature-wise min-max normalization to [0, 1]
        col_min = self.data.min(dim=0).values
        col_max = self.data.max(dim=0).values
        denom = torch.where((col_max - col_min) == 0, torch.ones_like(col_max), (col_max - col_min))
        self.data = (self.data - col_min) / denom
        
        logging.info(f"Loading shuttle {split}: {self.data.shape} from {file_path}")
    
    def _download_data(self, base_path: str) -> None:
        """Download shuttle dataset from UCI repository."""
        import zipfile
        import io
        import subprocess
        
        os.makedirs(base_path, exist_ok=True)
        
        # Download zip file
        zip_url = 'https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip'
        logging.info(f"Downloading shuttle data from {zip_url}")
        
        try:
            response = requests.get(zip_url)
            response.raise_for_status()
            
            # Extract the zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract files
                for name in z.namelist():
                    if 'shuttle.tst' in name:
                        # Test file is not compressed
                        content = z.read(name)
                        file_path = os.path.join(base_path, 'test.csv')
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        logging.info(f"Extracted {name} to {file_path}")
                    elif 'shuttle.trn.Z' in name:
                        # Training file is compressed with Unix compress (.Z)
                        compressed_content = z.read(name)
                        compressed_path = os.path.join(base_path, 'train.csv.Z')
                        with open(compressed_path, 'wb') as f:
                            f.write(compressed_content)
                        
                        # Decompress using uncompress command
                        try:
                            subprocess.run(['uncompress', compressed_path], check=True)
                            # Rename decompressed file
                            decompressed_path = os.path.join(base_path, 'train.csv')
                            if os.path.exists(decompressed_path.replace('.csv', '')):
                                os.rename(decompressed_path.replace('.csv', ''), decompressed_path)
                            logging.info(f"Decompressed and saved to {decompressed_path}")
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            # If uncompress is not available, try using Python's gzip/lzw
                            logging.warning("uncompress command not found, trying Python decompression")
                            # For now, skip this and use alternative approach
                            # Read as gzip might work
                            import gzip
                            try:
                                with open(compressed_path, 'rb') as f_in:
                                    content = gzip.decompress(f_in.read())
                                file_path = os.path.join(base_path, 'train.csv')
                                with open(file_path, 'wb') as f_out:
                                    f_out.write(content)
                                os.remove(compressed_path)
                                logging.info(f"Decompressed with gzip to {file_path}")
                            except:
                                # Last resort: use unlzw library if available
                                try:
                                    import unlzw
                                    with open(compressed_path, 'rb') as f:
                                        compressed = f.read()
                                    decompressed = unlzw.unlzw(compressed)
                                    file_path = os.path.join(base_path, 'train.csv')
                                    with open(file_path, 'wb') as f:
                                        f.write(decompressed)
                                    os.remove(compressed_path)
                                    logging.info(f"Decompressed with unlzw to {file_path}")
                                except ImportError:
                                    logging.error("Cannot decompress .Z file. Please install unlzw: pip install unlzw")
                                    raise
                        
        except Exception as e:
            logging.error(f"Failed to download shuttle dataset: {e}")
            logging.info("Please manually download from https://archive.ics.uci.edu/dataset/148/statlog+shuttle")
            raise
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx, :]
        return sample, idx


class LetterDataset(Dataset):
    """UCI Letter Recognition Dataset.

    16 numerical features per sample; labels are A-Z. Source file is CSV-like
    with 17 columns: class letter followed by 16 integers.
    """
    def __init__(self, split='train', data_path='') -> None:
        super(LetterDataset).__init__()
        self.split = split

        # Construct file path
        if data_path:
            base_path = os.path.join(data_path, 'data', 'letter')
        else:
            base_path = 'data/letter'

        # UCI provides a single file; use 80/20 split locally
        file_path = os.path.join(base_path, 'letter-recognition.data')

        if not os.path.exists(file_path):
            self._download_data(base_path)

        # Load raw data: first column is label, next 16 are numeric features
        df = pd.read_csv(file_path, header=None)
        # Expect 17 columns: [label, f1..f16]
        labels = df.iloc[:, 0].astype(str)
        features = df.iloc[:, 1:17].astype(float)

        # Normalize features feature-wise to [0, 1]
        feats = torch.tensor(features.values, dtype=torch.float32)
        col_min = feats.min(dim=0).values
        col_max = feats.max(dim=0).values
        denom = torch.where((col_max - col_min) == 0, torch.ones_like(col_max), (col_max - col_min))
        feats = (feats - col_min) / denom

        # Train/test split
        n = feats.shape[0]
        split_idx = int(0.8 * n)
        if split == 'train':
            self.data = feats[:split_idx]
            self.labels = labels.iloc[:split_idx].values
        elif split == 'test':
            self.data = feats[split_idx:]
            self.labels = labels.iloc[split_idx:].values
        else:
            self.data = feats
            self.labels = labels.values

        logging.info(f"Loading letter {split}: {self.data.shape} from {file_path}")

    def _download_data(self, base_path: str) -> None:
        os.makedirs(base_path, exist_ok=True)
        url = 'https://archive.ics.uci.edu/static/public/59/letter+recognition.zip'
        logging.info(f"Downloading letter dataset from {url}")
        import zipfile, io
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract letter-recognition.data (may be nested)
            target_name = 'letter-recognition.data'
            extracted = False
            for name in z.namelist():
                if name.endswith(target_name):
                    content = z.read(name)
                    out_path = os.path.join(base_path, target_name)
                    with open(out_path, 'wb') as f:
                        f.write(content)
                    logging.info(f"Extracted {name} to {out_path}")
                    extracted = True
                    break
            if not extracted:
                raise RuntimeError('letter-recognition.data not found in zip')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx, :]
        return sample, idx

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = tensor - min_tensor
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def create_batches(
    batch_size: int,
    dataset_name: str,
    data_path: str = "",
    shuffle_training: bool = True,
) -> Tuple[DataLoader, DataLoader, int, int]:
    # TODO improve dataset loading
    if dataset_name.startswith("binary_clustering"):
        # Dataset
        n_clusters, n_examples, n_dim = dataset_name.split("_")[-3:]
        train_dataset = BinaryClusterDataset(
            f"{data_path}binary_problem_data/train_T_{dataset_name}.pt"
        )
        test_dataset = BinaryClusterDataset(
            f"{data_path}binary_problem_data/test_T_{dataset_name}.pt"
        )
        width = int(n_dim)
        height = 1
    elif dataset_name == "msd":
        # TODO how to use datasets for paper comparisons
        train_dataset = MSDDataset("data/msd/pro_sg/train.csv")
        test_dataset = MSDDataset("data/msd/pro_sg/validation_tr.csv")
        assert train_dataset.get_n_dim() == test_dataset.get_n_dim()
        width = int(train_dataset.get_n_dim())
        height = 1
    elif dataset_name == "pendigits":
        train_dataset = PendigitsDataset(split='train', data_path=data_path)
        test_dataset = PendigitsDataset(split='test', data_path=data_path)
        width = 16  # pendigits has 16 features
        height = 1
    elif dataset_name == "shuttle":
        train_dataset = ShuttleDataset(split='train', data_path=data_path)
        test_dataset = ShuttleDataset(split='test', data_path=data_path)
        width = 9  # shuttle has 9 features
        height = 1
    elif dataset_name == "letter":
        train_dataset = LetterDataset(split='train', data_path=data_path)
        test_dataset = LetterDataset(split='test', data_path=data_path)
        width = 16  # letter recognition has 16 features
        height = 1

    else:
        raise Exception(f"Unknown data set: {dataset_name}")

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle_training
    )
    if not shuffle_training:
        assert train_loader.sampler.__class__.__name__ == "SequentialSampler"
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    logging.info(f"Create datasets with batch size {batch_size}")
    return train_loader, test_loader, width, height


def parse_arguments(param: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments (`sys.argv`).
    """
    parser = argparse.ArgumentParser(description="Data creation for Lipi-AE")
    (
        parser.add_argument(
            "--n_dim",
            type=int,
            default=100,
            help="Binary cluster data number of dimensions. E.g. 100",
        ),
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Binary cluster data number of dimensions. E.g. 10",
    )
    parser.add_argument(
        "--noise_probability",
        type=float,
        default=0.05,
        help="Noise probability. E.g. 0.05",
    )
    parser.add_argument(
        "--n_examples_per_cluster",
        type=int,
        default=100,
        help="Number of examples per cluster. E.g. 100",
    )
    args = parser.parse_args(param)
    return args


if __name__ == "__main__":
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    args = parse_arguments(sys.argv[1:])
    create_binary_cluster_data(
        n_dim=args.n_dim,
        n_clusters=args.n_clusters,
        noise_probability=args.noise_probability,
        n_examples_per_cluster=args.n_examples_per_cluster,
    )
