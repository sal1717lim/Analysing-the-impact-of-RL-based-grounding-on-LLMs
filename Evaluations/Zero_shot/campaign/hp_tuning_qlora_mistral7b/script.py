from itertools import product
import subprocess

HP = {
    'rl_script_args.lr': [1e-4, 1e-5, 1e-6],
    # 'rl_script_args.entropy_coef': [0.01, 0.05, 0.1],
    # 'rl_script_args.value_loss_coef': [0.25, 0.5],
    # 'rl_script_args.lora_r': [16, 32, 64],
    # 'rl_script_args.lora_alpha': [16, 32, 64]
}

SCRIPT = "/gpfswork/rech/imi/ucy39hi/Large-Scale_Grounding_LLMs_with_online_RL/experiments/campaign/GLAM_SCRIPT.slurm"
MODEL_TYPE = "causal"
# MODEL = "/gpfsscratch/rech/imi/ucy39hi/LLMs/Mistral-7B-v0.1"
MODEL = "/gpfsscratch/rech/imi/ucy39hi/LLMs/opt-6.7b"
TASK = "BabyAI-PickupLocal-v0"

base_script_run = ["sbatch", SCRIPT, MODEL_TYPE, MODEL, TASK]
runs = product(*[_v for _, _v in HP.items()])
n_runs = 0
for run in runs:
    n_runs += 1
    run_arguments = " ".join([f"{_k}={_v}" for _k, _v in zip(HP.keys(), run)])
    name = "HP_" + MODEL.split("/")[-1] + "_" + TASK + "_" + "-".join([f"{_k.split('.')[-1]}{_v}" for _k, _v in zip(HP.keys(), run)])
    subprocess.run(base_script_run + [name, run_arguments])

print("Total n runs: " + str(n_runs))

