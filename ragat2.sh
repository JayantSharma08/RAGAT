#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-0:0
#SBATCH --mem-per-cpu=2gb
#SBATCH --cpus-per-task=10
#SBATCH --gres gpu:1
#SBATCH -o myfile2.out
#SBATCH -w loewenburg150

module load CUDA/11.3.1
source activate ragat2

python run.py -epoch 800 -model ragat -score_func interacte -opn cross -gpu 0 -gcn_drop 0.4 -ifeat_drop 0.2 -ihid_drop 0.3 -batch 256 -iker_sz 11 -iperm 4 -attention True -head_num 1