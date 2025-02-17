#!/bin/bash
## SLURM Resource requirement:
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=LLM4AMR
#SBATCH --gres=gpu:v100:1
#SBATCH –output=myjob.%J.out
#SBATCH --error=myjob.%J.err
#SBATCH --time=10:00:00
## Required software list:

## Run the application:

# conda activate data
# echo "This job ran on $SLURM_NODELIST dated `date`";
python qwen_lora_amr.py &> output.txt

