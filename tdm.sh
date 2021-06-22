#!/bin/bash
#SBATCH --output=R-%x.%j.out                        # This allow to customize the output with file name and job ID
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a3090:1                        # 1 * 11GB GPU a3090, t8000, t2080ti
#SBATCH --mail-type=ALL                             # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=Salomon.Kabenamualu@tib.eu      # Email to which notifications will be sent

# bash code.sh
srun singularity exec --nv  /nfs/home/kabenamualus/Research/default.sif "$@"