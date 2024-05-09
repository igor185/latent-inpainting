export TORCH_HOME=/home/engineer/Dev/igor/thesis
export CUDA_VISIBLE_DEVICES=0,1
nohup python bin/train.py generator=ffc_resnet_075_mask_encoder run_title=lama-fourier-mask_encoder 2>&1 > log1.txt &!