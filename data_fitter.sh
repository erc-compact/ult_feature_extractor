#!/bin/bash

sbatch tests_fittings_csvmaker.sh /hercules/scratch/dbhatnagar/Classifier_data_samples/GBNCC_NP_pfds /hercules/scratch/dbhatnagar/Classifier_data_samples/24Oct_withsigma/gbncc
sbatch tests_fittings_csvmaker.sh /hercules/scratch/vishnu/RETRAINING_PROJECT/SURVEYS/GBNCC/pulsars /hercules/scratch/dbhatnagar/Classifier_data_samples/24Oct_withsigma/gbncc
sbatch tests_fittings_csvmaker.sh /hercules/scratch/vishnu/RETRAINING_PROJECT/SURVEYS/PALFA/pulsars /hercules/scratch/dbhatnagar/Classifier_data_samples/24Oct_withsigma/PALFA
sbatch tests_fittings_csvmaker.sh /hercules/scratch/vishnu/RETRAINING_PROJECT/SURVEYS/PALFA/nonpulsars /hercules/scratch/dbhatnagar/Classifier_data_samples/24Oct_withsigma/PALFA
sbatch tests_fittings_csvmaker.sh /hercules/scratch/vishnu/RETRAINING_PROJECT/SURVEYS/FAST/PICS-ResNet_data/train_data/pulsar /hercules/scratch/dbhatnagar/Classifier_data_samples/24Oct_withsigma/fast
sbatch tests_fittings_csvmaker.sh /hercules/scratch/vishnu/RETRAINING_PROJECT/SURVEYS/FAST/PICS-ResNet_data/train_data/rfi /hercules/scratch/dbhatnagar/Classifier_data_samples/24Oct_withsigma/fast
sbatch tests_fittings_csvmaker.sh /hercules/scratch/dbhatnagar/LOWLAT_data/mini_dataset/pulsar/pulsar_only /hercules/scratch/dbhatnagar/Classifier_data_samples/24Oct_withsigma/lowlat
sbatch tests_fittings_csvmaker.sh /hercules/scratch/dbhatnagar/LOWLAT_data/mini_dataset/nonpulsar_only /hercules/scratch/dbhatnagar/Classifier_data_samples/24Oct_withsigma/lowlat


