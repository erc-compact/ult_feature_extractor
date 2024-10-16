#!/bin/bash
#SBATCH --partition=short.q
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=7G
#SBATCH --time=4:00:00
#SBATCH --job-name=extract

tmp_dir="/tmp/dbhatnagar"

rm -rf $tmp_dir
mkdir -p $tmp_dir

#path="/hercules/scratch/vishnu/RETRAINING_PROJECT/SURVEYS/PALFA/nonpulsars"
path=$1
# tag=$2
# OD=$3

sing_img="/hercules/:/hercules/ /hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/pulsar-miner_plot_dm_off.sif"

singularity exec -H $HOME:/home -B $sing_img python3 all_fits.py --dir "$path" 

#output_dir="/hercules/scratch/dbhatnagar/Classifier_data_samples/PALFA_chis"
output_dir=$2
rsync -Pav  $tmp_dir/*.csv  $output_dir
rm -rf $tmp_dir