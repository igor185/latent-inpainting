export TORCH_HOME=/home/engineer/Dev/igor/thesis
export CUDA_VISIBLE_DEVICES=0
nohup python bin/train.py -cn lama-fourier-latent-l2 run_title=lama-fourier-latent-full-l2 generator=ffc_resnet_075_autoencoder_8_latentl2 2>&1 > log0.txt &!