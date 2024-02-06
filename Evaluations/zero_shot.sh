#!/bin/bash
#SBATCH --partition=hard
#SBATCH --nodelist=thin
#SBATCH --job-name=zeroshot

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1


#SBATCH --output=%x-%j.out

#SBATCH --error=%x-%j.err
export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"

python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=0
python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=1
python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=2
python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=3
python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=4
python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=5
python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=6
python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=7
python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=8
python3 -m lamorel_launcher.launch --config-path "/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/configs" --config-name "local_gpu_config"        rl_script_args.path="/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/zero_shot.py" rl_script_args.output_dir="." lamorel_args.accelerate_args.machine_rank=0 lamorel_args.llm_args.model_name="google/flan-t5-large" lamorel_args.llm_args.model_path="google/flan-t5-large" lamorel_args.llm_args.prompt_number=9


