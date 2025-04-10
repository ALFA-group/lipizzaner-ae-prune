# Lipizzaner Autoencoder (Lipi-Ae)

## Installation

Create virtual environment. E.g
```
python3 -m venv ~/.venvs/lipi_ae_gecco_24
```

Activate virtual environment. E.g
```
source ~/.venvs/lipi_ae_gecco_24/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

## Quick start

##### GECCO 2025 experiment runs

Create data.
```
PYTHONPATH=src python src/aes_lipi/utilities/gecco_25_experiments.py --create_data True
```

Run experiments. **Note** Reduces the amount of config files, but script needs to be changed
```
 time PYTHONPATH=src python src/aes_lipi/utilities/gecco_25_experiments.py --rq <RQ_NAME>
```

### Binary Clustering

Create Binary Clustering problems
```
PYTHONPATH=src python src/aes_lipi/datasets/data_loader.py --n_dim 1000 --n_clusters 10
```

Test binary clustering problem and autoencoder
```
PYTHONPATH=src python src/aes_lipi/environments/binary_clustering.py --method=Autoencoder --dataset_name=binary_clustering_10_100_1000
```

#### Lipi-AE variants

##### Lipi-AE update selected and random - GECCO 2024 variant
Run Lipi-Ae on binary clustering problem
```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --configuration_file=tests/gecco_2024/configurations/binary_clustering/test_bc/binary_clustering_epoch_node_demo_lipi_ae.json
```

##### Lipi-AE update only selected - Simple variant
Run Lipi-Ae simple variant on binary clustering problem
```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --dataset_name binary_clustering_10_100_1000  --environment AutoencoderBinaryClustering --epochs 3 --batch_size 400 --population_size 3 --ae_quality_measures L1 --cell_evaluation lipi_simple --output_dir out_lipi_simple --no_output_dir_timestamp True
```

### Activation analysis

Save the parameters at every iteration
Run Lipi-Ae with solution concept `best_case`
```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --dataset_name binary_clustering_10_100_1000  --environment AutoencoderBinaryClustering --epochs 3 --batch_size 3 --population_size 3 --ae_quality_measures L1 --checkpoint_interval 1 --do_not_overwrite_checkpoint --output_dir out_activation_analysis --no_output_dir_timestamp True
```

Analyse the activations and weights of ANNs
```
PYTHONPATH=src python src/aes_lipi/utilities/activation_analysis.py --data_prefix out_activation_analysis
```

##### Actication at every update step and deterministic data order

Save the parameters at every update `--checkpoint_every_update` and fixed data order `--no_shuffle_data`
```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --dataset_name binary_clustering_10_100_1000  --environment AutoencoderBinaryClustering --epochs 3 --batch_size 3 --population_size 3 --ae_quality_measures L1 --output_dir checkpoint_every_update --checkpoint_every_update True --no_shuffle_data True --no_output_dir_timestamp True
```

Analyse the activations and weights of ANNs
```
PYTHONPATH=src python src/aes_lipi/utilities/activation_analysis.py --data_prefix checkpoint_every_update
```

#### AE architectures

Try small and large architecture with `--environment AutoencoderBinaryClustering_Small` and `--environment AutoencoderBinaryClustering_Large`. E.g.

```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --dataset_name binary_clustering_10_100_1000  --environment AutoencoderBinaryClustering_Small --epochs 3 --batch_size 3 --population_size 3 --ae_quality_measures L1 --output_dir prune_ann --no_output_dir_timestamp True --prune_probability 1.0 --prune_amount 0.1 --prune_method activation --cell_evaluation lipi_simple
```

#### Prune Networks
Prune a network with
```
PYTHONPATH=src python src/aes_lipi/anns/prune_ann.py --dataset_name binary_clustering_10_100_1000  --method Autoencoder
```

##### Evolve with pruning
Pruning probability `--prune_probability`, prune amount `--prune_amount`, prune method `--prune_method`, and prune schedule `--prune_schedule`
```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --dataset_name binary_clustering_10_100_1000  --environment AutoencoderBinaryClustering --epochs 3 --batch_size 3 --population_size 3 --ae_quality_measures L1 --output_dir prune_ann --no_output_dir_timestamp True --prune_probability 1.0 --prune_amount 0.1 --prune_method activation --prune_schedule increase --cell_evaluation lipi_simple
```


#### Experiment

##### GECCO 2025 experiment runs

Create data.
```
PYTHONPATH=src python src/aes_lipi/utilities/gecco_25_experiments.py --create_data True
```

Run experiments with test variations `--rq`. **Note** Reduces the amount of config files, but script needs to be changed
```
 time PYTHONPATH=src python src/aes_lipi/utilities/gecco_25_experiments.py --rq test
```

Get CLI commands
```
 time PYTHONPATH=src python src/aes_lipi/utilities/gecco_25_experiments.py --print_cli_only > settings_for_experiment.sh
```

Then run with:
```
bash ./settings_for_experiment.sh
```

##### General experiment runs
Run experiments
```
 time PYTHONPATH=src python src/aes_lipi/utilities/gecco_experiments.py --configuration_directory tests/gecco_2024/configurations/binary_clustering/test_bc --sensitivity tests/gecco_2024/configurations/binary_clustering/test_bc/sensitivity_values.json
```

Update dataset in `sensitivity_values.json` key `"dataset_name"` by adding the new dataset to the list

Analyze data from `--root_dir` based on `--param_dir` parameters.
```
time PYTHONPATH=src python src/aes_lipi/utilities/analyse_data.py --root_dir out_binary_clustering --param_dir out_binary_clustering 
```

##### Compare ANN parameters

Save the parameters at every iteration
Run Lipi-Ae
```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --dataset_name binary_clustering_10_100_1000  --environment AutoencoderBinaryClustering --epochs 3 --batch_size 400 --population_size 3 --ae_quality_measures L1 --checkpoint_interval 1 --do_not_overwrite_checkpoint
```

#### Small Activation analysis

Create Binary Clustering problems
```
PYTHONPATH=src python src/aes_lipi/datasets/data_loader.py --n_dim 10 --n_clusters 2
```

Test binary clustering problem and autoencoder
```
PYTHONPATH=src python src/aes_lipi/environments/binary_clustering.py --method=Autoencoder --dataset_name=binary_clustering_2_100_10
```

Save the parameters at every iteration
Run Lipi-Ae w
```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --dataset_name binary_clustering_2_100_10  --environment AutoencoderBinaryClustering --epochs 3 --batch_size 40 --population_size 3 --ae_quality_measures L1 --checkpoint_interval 1 --do_not_overwrite_checkpoint --output_dir out_activation_analysis_small
```

Analyse the activations and weights of ANNs
```
PYTHONPATH=src python src/aes_lipi/utilities/activation_analysis.py --data_prefix out_activation_analysis_small --dataset_name binary_clustering_2_100_10
```

### Experiments

**TODO** Simplified or complex lipi. Well a good test and contribution for the paper

`src/aes_lipi/utilities/gecco_25_experiments.py` has experiments settings.

#### Store all networks before updating weights

**Note** This will store a lot of networks. 

```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --dataset_name binary_clustering_2_100_10  --environment AutoencoderBinaryClustering --epochs 3 --batch_size 40 --population_size 3 --ae_quality_measures L1 --checkpoint_interval 1 --do_not_overwrite_checkpoint --output_dir out_activation_analysis_small --checkpoint_all_networks_before_updating_weights
```

## Reference

```

@inproceedings{hemberg2024ae,
  title={Cooperative Spatial Topologies for Autoencoder Training},
  author={Hemberg, Erik and Toutouh, Jamal and O'Reilly, Una-May},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  year={2024}
}
```
