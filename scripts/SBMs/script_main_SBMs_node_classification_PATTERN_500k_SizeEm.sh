#!/bin/bash
code=main_SBMs_node_classification.py 
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --lap_pos_enc false --batch_size 20 --epochs 20
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --pos_enc_dim 1 --batch_size 20 --epochs 20
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --pos_enc_dim 2 --batch_size 20 --epochs 20
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --pos_enc_dim 5 --batch_size 20 --epochs 16
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --pos_enc_dim 10 --batch_size 20 --epochs 16
python $code --config 'configs/SBMs_GraphTransformer_LapPE_PATTERN_500k_sparse_graph_BN.json' --pos_enc_dim 30 --batch_size 20 --epochs 16
git add -A
git commit -m "Run modification pos_enc_dim"
git push
