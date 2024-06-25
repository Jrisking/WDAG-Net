#PBS -S /bin/bash
#PBS -o script.out
#PBS -j oe
#PBS -l select=1:ncpus=8
cd $PBS_O_WORKDIR
/home/hoo/.conda/envs/ljh/bin/python /home/ljh/ljh/PythonProject/WDAGNet/Train.py