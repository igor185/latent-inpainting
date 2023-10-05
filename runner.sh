export PYTHONPATH=/home/engineer/Dev/igor/lama-custom
export TORCH_HOME=/home/engineer/Dev/igor/lama-custom
nohup python bin/train.py -cn lama-fourier-sd location=places_example 2>&1 > log.txt &!