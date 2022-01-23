#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=thin 
#SBATCH --time=08:00:00
#
python optimisation_Cabauw.py


