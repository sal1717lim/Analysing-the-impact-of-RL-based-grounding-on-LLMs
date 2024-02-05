#!/bin/bash
#SBATCH --job-name=BabyAIPickup_LoRA_FlanT5780M_%a    # job name
#SBATCH --time=20:00:00 # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/BabyAIPickup_LoRA_FlanT5780M_%a-%j.out     # output file name
#SBATCH --error=slurm_logs/BabyAIPickup_LoRA_FlanT5780M_%a-%j.err      # err file name
#SBATCH --account=imi@a100
#SBATCH --qos=qos_gpu-t3
#SBATCH -C a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

#SBATCH --array=1-1

module purge
module load cpuarch/amd
module load python/3.8.2
conda activate ls_grounding_llms

chmod +x experiments/campaign/lamorel_launcher.sh

srun experiments/campaign/lamorel_launcher.sh \
                    rl_script_args.path=$WORK/Large-Scale_Grounding_LLMs_with_online_RL/experiments/main.py \
                    rl_script_args.seed=${SLURM_ARRAY_TASK_ID} \
                    rl_script_args.number_envs=32 \
                    rl_script_args.epochs=1000 \
                    rl_script_args.steps_per_epoch=1280 \
                    rl_script_args.output_dir=$SCRATCH/LS_Grounding_LLMs/BabyAIPickupLocal_FlanT5780M_LoRA/seed_${SLURM_ARRAY_TASK_ID} \
                    rl_script_args.save_freq=10 \
                    rl_script_args.name_environment='babyai_text' \
                    rl_script_args.task='BabyAI-PickupLocal-v0' \
                    rl_script_args.minibatch_size=256 \
                    rl_script_args.gradient_batch_size=8 \
                    rl_script_args.use_lora=true \
                    rl_script_args.lr=1e-4 \
                    rl_script_args.use_all_params_for_optim=false \
                    rl_script_args.quantized_optimizer=false \
                    lamorel_args.llm_args.model_type=seq2seq \
                    lamorel_args.llm_args.model_path=$SCRATCH/LLMs/flan-t5-large \
                    lamorel_args.llm_args.load_in_4bit=false \
                    lamorel_args.llm_args.parallelism.model_parallelism_size=1 \
                    lamorel_args.llm_args.parallelism.empty_cuda_cache_after_scoring=false \
                    lamorel_args.llm_args.minibatch_size=192 \
                    lamorel_args.accelerate_args.num_machines=1 \
                    --config-path=$WORK/Large-Scale_Grounding_LLMs_with_online_RL/experiments/configs \
                    --config-name=multi-node_slurm_cluster_config 
