export TORCH_HOME=/home/engineer/Dev/igor/thesis
export CUDA_VISIBLE_DEVICES=0,1
nohup python bin/train.py 2>&1 > log.txt &!