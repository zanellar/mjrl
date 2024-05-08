#!/bin/bash
# parameters for slurm
#SBATCH -J mjrl                   # job name, don't use spaces, keep it short 
#SBATCH --mail-type=FAIL          # email status changes (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=r.zanella@utwente.nl   # Where to send mail to
#SBATCH --time=90-00:00                # time limit 1h
#SBATCH --output=job_test_%j.log      # Standard output and error log 
 
module load anaconda3/2022.05
source activate
conda activate mjrl
python ../sweep/sweep_panda_posctrl_posreach_energylimit1.py