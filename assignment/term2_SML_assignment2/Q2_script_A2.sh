#!/bin/bash
#$ -l h_rt=10:00:00  #time needed
#$ -pe smp 20 #number of cores
#$ -l rmem=1G #number of memery
#$ -P rse-com6012
#$ -q rse-com6012.q
#$ -o Q2_A2.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M yzhang320@sheffield.ac.uk
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 20g Code/Q2_code_A2.py