export TORCH_HOME=/home/engineer/Dev/igor/thesis
export CUDA_VISIBLE_DEVICES=0,1
nohup python bin/train.py -cn lama-fourier-latent-l2 run_title=lama-fourier-autoencoder-8-latent-l2 generator=ffc_resnet_075_autoencoder_8_latentl2 2>&1 > log2.txt &!