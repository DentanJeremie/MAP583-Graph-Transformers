#!/bin/bash
code=main_SBMs_node_classification.py 
python $code --config 'configs/ablation_SBMs_GraphTransformer_WLPE_PATTERN_500k_sparse_graph_BN.json' --renormalization_pos_enc 0.2
python "print(\n\n\n\nNEW RUN\n\n\n\n)"
python $code --config 'configs/ablation_SBMs_GraphTransformer_WLPE_PATTERN_500k_sparse_graph_BN.json' --renormalization_pos_enc 0.1