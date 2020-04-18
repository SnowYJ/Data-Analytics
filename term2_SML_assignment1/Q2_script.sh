#!/bin/bash
#$ -l h_rt=2:00:00  #time needed
#$ -pe smp 3 #number of cores
#$ -l rmem=4G #number of memery
#$ -o Q2.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M yzhang320@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit Code/Q2_code.py
