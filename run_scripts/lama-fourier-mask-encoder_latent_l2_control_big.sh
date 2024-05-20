export TORCH_HOME=/home/engineer/Dev/igor/thesis
export CUDA_VISIBLE_DEVICES=2,3
nohup python bin/train.py -cn lama-fourier generator=ffc_resnet_075_autoencoder_8_latentl2 run_title=lama-fourier-mask-control-full-big 2>&1 > log2.txt &!