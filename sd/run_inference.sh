export CUDA_VISIBLE_DEVICES=3
nohup python scripts/inpaint.py --indir /mnt/data/datasets/places_standard_dataset/evaluation 2>&1 > log1.txt &!