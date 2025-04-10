import os
import unittest
import pathlib

from aes_lipi.lipi_ae import main


class TestLipiAE(unittest.TestCase):
    # TODO test cli args

    def test_main(self):
        params = {
            "configuration_file": "None",
            "visualize": "none",
            "dataset_name": "binary_clustering_10_100_1000",
            "checkpoint_interval": 1,
            "rng_seed": 1,
            "radius": 1,
            "ann_path": "",
            "cell_evaluation": "epoch-node",
            "ae_quality_measures": "L1",
            "output_dir": "out_unittest",
            "environment": "AutoencoderBinaryClustering",
            "log_level": "info",
            "solution_concept": "best_case",
            "population_size": 3,
            "epochs": 3,
            "batch_size": 400,
            "learning_rate": 1e-05,
            "no_execution": False,
            "overwrite_checkpoint": False,
            "store_all_nodes": True,
            "calculate_test_loss": False,
            "checkpoint_every_update": True,
            "no_shuffle_data": True,
            "timestamp": 1734798122.147885,
        }
        main(**params)
        checkpoint_dir = pathlib.Path(params["output_dir"]) / "checkpoints"
        print(os.listdir(checkpoint_dir))
