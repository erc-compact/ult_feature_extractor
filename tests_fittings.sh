#!/bin/bash
file=$1
#config=$2
#sing_img="/hercules/:/hercules/ /hercules/scratch/vishnu/singularity_images/pulsar-miner_turing-sm75.sif"
sing_img="/hercules/:/hercules/ /hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/pulsar-miner_plot_dm_off.sif"
#command1="python3 all_features_presto.py --file \"$file\""
#command1="python3 features_main.py --file \"$file\" --config  \"$config\""
#singularity exec -H $HOME:/home -B $sing_img $command1 
singularity exec -H $HOME:/home -B $sing_img python3 fittings_main.py --file "$file" 