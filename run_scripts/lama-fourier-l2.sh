export TORCH_HOME=/home/engineer/Dev/igor/thesis
export CUDA_VISIBLE_DEVICES=3
nohup python bin/train.py -cn lama-fourier-l2 run_title=lama-fourier-l2 location=places generator=ffc_resnet_075_8 2>&1 > log2.txt &!