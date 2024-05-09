export TORCH_HOME=/home/engineer/Dev/igor/thesis
export CUDA_VISIBLE_DEVICES=2,3
nohup python bin/train.py -cn lama-fourier run_title=lama-fourier-autoencoder-8 generator=ffc_resnet_075_autoencoder_8 2>&1 > log.txt &!