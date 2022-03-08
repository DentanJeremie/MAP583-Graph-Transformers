

# Graph Transformer Architecture

This work is based on source code of the paper "**[A Generalization of Transformer Networks to Graphs](https://arxiv.org/abs/2012.09699)**" by _[Vijay Prakash Dwivedi](https://github.com/vijaydwivedi75) and [Xavier Bresson](https://github.com/xbresson)_, at **AAAI'21 Workshop on Deep Learning on Graphs: Methods and Applications (DLG-AAAI'21)**. 

## I. Adaptations made to the source code

Our adaptations are the following :

- New scripts to conduct other experiment.
- New arguments available with `main_SBMs_node_classification.py`. Some of them enable to save to model to test it on other dataset, some of them enable to add a renormalization of the laplacian encoding.
- Creation of `test_SBM.py` to test a saved model on a specific dataset
- Two jupyter notebook in the folder `data/SBMs/generate_datasets`

The result of our adaptations are presented in the `presentation` folder.

### 1. New scripts

Two new scripts are available :
```
bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN_500k_LaplRenorm.sh

bash scripts/SBMs/ script_main_SBMs_node_classification_PATTERN_500k_SizeEm.sh
```

- The first one provides empirical results on the effect of an homothety applied to the laplacian encoding.
- The second one provides empirical results on the effect of changing the dimension of the laplacian encoding (which means taking more or less eigenvectors of the laplacian for the embedding).

### 2. New arguments for `main_SBMs_node_classification.py`

- New arguments for homothety : the argument `--renormalization_pos_enc`has been added to the argument parser of `main_SBMs_node_classification.py`. This argument is the factor of the homothety applied to the laplacian encoding, and overwrites the one in the config file if provided.
- New argument to save the model or load one : this enables to save a model (either to train if again later or to test it on another dataset) and to load it to continue the training phase. Usage :

```
python main_SBMs_node_classification.py --save_model out/models/modelName

python main_SBMs --load_model out/models/modelName
```

Two pretrained models are provided in out/models (80 epoch, about about 3h execution time for each). Their configuration are the following :

For `baseSMB-lapEnc` :
```
Dataset: SBM_PATTERN,
Model: GraphTransformer

params={'seed': 41, 'epochs': 80, 'batch_size': 20, 'init_lr': 0.0005, 'lr_reduce_factor': 0.5, 'lr_schedule_patience': 10, 'min_lr': 1e-06, 'weight_decay': 0.0, 'print_epoch_interval': 5, 'max_time': 24}

net_params={'L': 10, 'n_heads': 8, 'hidden_dim': 80, 'out_dim': 80, 'residual': True, 'readout': 'mean', 'in_feat_dropout': 0.0, 'dropout': 0.0, 'layer_norm': False, 'batch_norm': True, 'self_loop': False, 'lap_pos_enc': True, 'pos_enc_dim': 2, 'wl_pos_enc': False, 'full_graph': False, 'device': device(type='cuda'), 'gpu_id': 0, 'batch_size': 20, 'renormalization_pos_enc': 1.0, 'in_dim': 3, 'n_classes': 2, 'total_param': 522982}


Total Parameters: 522982
```

For `baseBSM-NoLap` :
```
Dataset: SBM_PATTERN,
Model: GraphTransformer

params={'seed': 41, 'epochs': 80, 'batch_size': 20, 'init_lr': 0.0005, 'lr_reduce_factor': 0.5, 'lr_schedule_patience': 10, 'min_lr': 1e-06, 'weight_decay': 0.0, 'print_epoch_interval': 5, 'max_time': 24}

net_params={'L': 10, 'n_heads': 8, 'hidden_dim': 80, 'out_dim': 80, 'residual': True, 'readout': 'mean', 'in_feat_dropout': 0.0, 'dropout': 0.0, 'layer_norm': False, 'batch_norm': True, 'self_loop': False, 'lap_pos_enc': False, 'pos_enc_dim': 2, 'wl_pos_enc': False, 'full_graph': False, 'device': device(type='cuda'), 'gpu_id': 0, 'batch_size': 20, 'renormalization_pos_enc': 1.0, 'in_dim': 3, 'n_classes': 2, 'total_param': 522742}


Total Parameters: 522742
```

### 3. Creation of a test file

A test file `test_SBM.py` has been created to test a model (saved as described previously) on another dataset. Usage :

```
python test_SBM.py --load_model out/models/modelName --config config/configName --dataset data/SBMs/datasetName
```

### 4. Two jupyter notebook to generate the dataset

This is a long operation (>5h). The two jupyter notebooks are in `data/SBMs/generate_datasets` First execute `generate_SBM_PATTERN.ipynb`and then `prepare_SBM_PATTERN.ipynb`.

## II. Readme content of the source code of the paper

Source code for the paper "**[A Generalization of Transformer Networks to Graphs](https://arxiv.org/abs/2012.09699)**" by _[Vijay Prakash Dwivedi](https://github.com/vijaydwivedi75) and [Xavier Bresson](https://github.com/xbresson)_, at **AAAI'21 Workshop on Deep Learning on Graphs: Methods and Applications (DLG-AAAI'21)**.

We propose a generalization of transformer neural network architecture for arbitrary graphs: **Graph Transformer**. <br>Compared to the [Standard Transformer](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), the highlights of the presented architecture are: 

- The attention mechanism is a function of neighborhood connectivity for each node in the graph.  
- The position encoding is represented by Laplacian eigenvectors, which naturally generalize the sinusoidal positional encodings often used in NLP.  
- The layer normalization is replaced by a batch normalization layer.  
- The architecture is extended to have edge representation, which can be critical to tasks with rich information on the edges, or pairwise interactions (such as bond types in molecules, or relationship type in KGs. etc). 

<br>

<p align="center">
  <img src="./docs/graph_transformer.png" alt="Graph Transformer Architecture" width="800">
  <br>
  <b>Figure</b>: Block Diagram of Graph Transformer Architecture
</p>


### 1. Repo installation

This project is based on the [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns) repository.

[Follow these instructions](./docs/01_benchmark_installation.md) to install the benchmark and setup the environment.


<br>

### 2. Download datasets

[Proceed as follows](./docs/02_download_datasets.md) to download the datasets used to evaluate Graph Transformer.


<br>

### 3. Reproducibility 

[Use this page](./docs/03_run_codes.md) to run the codes and reproduce the published results.


<br>

### 4. Reference 

:page_with_curl: Paper [on arXiv](https://arxiv.org/abs/2012.09699)    
:pencil: Blog [on Towards Data Science](https://towardsdatascience.com/graph-transformer-generalization-of-transformers-to-graphs-ead2448cff8b)    
:movie_camera: Video [on YouTube](https://www.youtube.com/watch?v=h-_HNeBmaaU&t=237s)    
```
@article{dwivedi2021generalization,
  title={A Generalization of Transformer Networks to Graphs},
  author={Dwivedi, Vijay Prakash and Bresson, Xavier},
  journal={AAAI Workshop on Deep Learning on Graphs: Methods and Applications},
  year={2021}
}
```

<br><br><br>

