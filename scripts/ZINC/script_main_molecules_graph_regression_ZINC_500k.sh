#!/bin/bash


############
# Usage
############

# bash script_main_xx.sh


############
# ZINC - 4 RUNS
############

seed0=41
seed1=95
seed2=22
seed3=35
code=main_molecules_graph_regression.py 
dataset=ZINC
tmux new -s screen_GT -d
tmux send-keys "source activate graph_transformer" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_GraphTransformer_ZINC_500k_sparse_graph_LN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_GraphTransformer_ZINC_500k_sparse_graph_LN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_GraphTransformer_ZINC_500k_sparse_graph_LN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_GraphTransformer_ZINC_500k_sparse_graph_LN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_LN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_LN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_LN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_LN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_GraphTransformer_ZINC_500k_full_graph_LN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_GraphTransformer_ZINC_500k_full_graph_LN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_GraphTransformer_ZINC_500k_full_graph_LN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_GraphTransformer_ZINC_500k_full_graph_LN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_full_graph_LN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_full_graph_LN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_full_graph_LN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_full_graph_LN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_GraphTransformer_ZINC_500k_sparse_graph_BN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_GraphTransformer_ZINC_500k_sparse_graph_BN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_GraphTransformer_ZINC_500k_sparse_graph_BN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_GraphTransformer_ZINC_500k_sparse_graph_BN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_GraphTransformer_ZINC_500k_full_graph_BN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_GraphTransformer_ZINC_500k_full_graph_BN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_GraphTransformer_ZINC_500k_full_graph_BN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_GraphTransformer_ZINC_500k_full_graph_BN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_full_graph_BN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_full_graph_BN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_full_graph_BN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_full_graph_BN.json' &
wait" C-m
tmux send-keys "tmux kill-session -t screen_GT" C-m











