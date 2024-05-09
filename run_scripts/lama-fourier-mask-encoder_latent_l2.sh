export TORCH_HOME=/home/engineer/Dev/igor/thesis
export CUDA_VISIBLE_DEVICES=3
nohup python bin/train.py -cn lama-fourier-latent-l2 generator=ffc_resnet_075_mask_encoder_latent run_title=lama-fourier-mask_encoder_latent_l2 2>&1 > log2.txt &!