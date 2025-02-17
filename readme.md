# LLM for AMR Prediction

- Step1. create a conda environment: 
    conda create -n llm4amr python==3.9
    conda activate llm4amr

- Step2. install required packages: 
    pip install -r requirements.txt

- Step3: submit the SLURM task to run the training codes
    - cd LLM4AMR # working dir
    - run the SLURM task:
        sbatch sbatch_LLM_arm.sh

