#!/bin/bash
code=main_SBMs_node_classification.py 
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --renormalization_pos_enc 0.01
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --renormalization_pos_enc 0.3
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --renormalization_pos_enc 0.7
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --renormalization_pos_enc 5.0
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --renormalization_pos_enc 20.0
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --renormalization_pos_enc 100.0