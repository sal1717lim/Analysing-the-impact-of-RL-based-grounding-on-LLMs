'''
PPO implementation taken from https://github.com/openai/spinningup
'''
from collections import OrderedDict
from typing import List

import hydra
from utils.ppo_buffer import PPOBuffer
from utils.generate_prompt import generate_prompt
from utils.generate_prompt import *
from utils.scoring_utils import scores_stacking
import torch
import bitsandbytes
import numpy as np
import logging

nombre_episode=300
from transformers import set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from tqdm import tqdm
import time
import pickle
import math
import os
import functools as f
from operator import add

import gym
from environments import EnvEnum

from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction, BaseModelInitializer

lamorel_init()
from babyai.bot import Bot
from accelerate import Accelerator

accelerator = Accelerator()

class LogScoringModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"]) # inputs are padded so all of same size

            logits = forward_outputs["logits"][:, end_of_context_position:-1, :]
            output_tokens = minibatch["input_ids"][:, end_of_context_position+1:]
        else:
            logits = forward_outputs["logits"][:, :-1, :]  # skip </s> token appended by tokenizer
            output_tokens = minibatch["decoder_input_ids"][:, 1:]  # skip pad token

        tokens_logprobs = \
            torch.gather(logits, 2, output_tokens[:, :, None]).squeeze(-1).to(torch.float32)  # filter with sequence tokens

        # Compute mask to assign probability 1 to padding tokens
        mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_logprobs.masked_fill(mask, 0.0)  # apply mask
        minibatch_probs = masked_token_probs.sum(-1)  # compute final sequences' probability

        return minibatch_probs.cpu()

class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        if 'hidden_size' in self.llm_config.attribute_map:
            _hidden_size_key = self.llm_config.attribute_map['hidden_size']
        else:
            if "word_embed_proj_dim" in self.llm_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.llm_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.llm_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")

        self._llm_hidden_size = self.llm_config.to_dict()[_hidden_size_key]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"])  # inputs are padded so all of same size

            model_head = forward_outputs['hidden_states'][-1][:, end_of_context_position, :]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]

        value = self.value_head_op(model_head.to(torch.float32).to(self.device))
        return value.cpu()

class SequentialInitializer(BaseModelInitializer):
    def __init__(self, initializers:List[BaseModelInitializer]):
        super().__init__()
        self._initializers = initializers

    def initialize_model(self, model):
        for _initializer in self._initializers:
            model = _initializer.initialize_model(model)

        return model

class WeightsLoaderInitializer(BaseModelInitializer):
    def __init__(self, weights_path):
        super().__init__()
        self._weights_path = weights_path

    def initialize_model(self, model):
        if self._weights_path is not None:
            loaded_ddp_dict = torch.load(self._weights_path + "/model.checkpoint",map_location=torch.device('cpu'))
            hf_llm_module_dict = {_k.replace('module.', ''): _v for _k, _v in loaded_ddp_dict.items()}
            model.load_state_dict(state_dict=hf_llm_module_dict, strict=False)

        return model

class PeftInitializer(BaseModelInitializer):
    def __init__(self, model_type, model_name, use_lora, use_4bit, r, alpha, use_cache=True):
        super().__init__()
        self._model_type = model_type
        self._model_name = model_name
        self._use_lora = use_lora
        self._use_4bit = use_4bit
        self._r = r
        self._alpha = alpha
        self._use_cache = use_cache

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _get_model_config(self):
        if "t5" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q", "v"],
                lora_dropout=0.0,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
        elif "falcon" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=[
                    "query_key_value",
                    "dense",
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                ],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
        elif "opt" in self._model_name or "Llama" in self._model_name or "Mistral" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
        else:
            raise NotImplementedError()

    def initialize_model(self, model):
        if self._use_lora:
            llm_module = model._modules['_LLM_model']
            if self._model_type == "seq2seq" or not self._use_cache:
                llm_module.gradient_checkpointing_enable()  # reduce number of stored activations

            if self._use_4bit:
                llm_module = prepare_model_for_kbit_training(llm_module)

            # Init adapters #
            config = self._get_model_config()
            peft_model = get_peft_model(llm_module, config)
            parent_module_device = None
            for name, param in peft_model.named_modules():
                if name.split(".")[-1].startswith("lora_"):
                    if hasattr(param, "weight"):
                        param.to(parent_module_device)
                else:
                    if hasattr(param, "weight"):
                        parent_module_device = param.weight.device
                    else:
                        parent_module_device = None

            model._modules['_LLM_model'] = peft_model

        model.eval()  # Important to ensure ratios are 1 in first minibatch of PPO (i.e. no dropout)
        model._modules['_LLM_model'].config.use_cache = self._use_cache
        self._print_trainable_parameters(model)
        return model

class PPOUpdater(BaseUpdater):
    def __init__(self, model_type, minibatch_size, gradient_batch_size, quantized_optimizer, gradient_minibatch_size=None):
        super(PPOUpdater, self).__init__()
        self._model_type = model_type
        self._minibatch_size = minibatch_size
        self._gradient_batch_size = gradient_batch_size
        self._gradient_minibatch_size = gradient_minibatch_size
        self._quantized_optimizer = quantized_optimizer

    def _get_trainable_params(self, model, return_with_names=False):
        if return_with_names:
            return filter(lambda p: p[1].requires_grad, model.named_parameters())
        else:
            return filter(lambda p: p.requires_grad, model.parameters())

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'optimizer'):
            if kwargs["use_all_params_for_optim"]:
                self._iterator_named_trainable_params = self._llm_module.named_parameters
            else:
                self._iterator_named_trainable_params = lambda: self._get_trainable_params(self._llm_module, True)

            self._iterator_trainable_params = (p for n, p in self._iterator_named_trainable_params())
            if self._quantized_optimizer:
                self.optimizer = bitsandbytes.optim.PagedAdamW8bit(self._iterator_trainable_params, lr=kwargs["lr"])
            else:
                self.optimizer = torch.optim.Adam(self._iterator_trainable_params, lr=kwargs["lr"])

            if os.path.exists(kwargs["loading_path"] + "/optimizer.checkpoint"):
                self.optimizer.load_state_dict(torch.load(kwargs["loading_path"] + "/optimizer.checkpoint"))

        current_process_buffer = {}
        for k in ['actions', 'advantages', 'returns', 'logprobs', 'values']:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        epochs_losses = {
            "value": [],
            "policy": [],
            "loss": []
        }

        n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
            for step in range(n_minibatches):
                _minibatch_start_idx = step * self._minibatch_size
                _minibatch_end_idx = min(
                    (step + 1) * self._minibatch_size,
                    len(contexts))

                self.optimizer.zero_grad()
                gradient_accumulation_steps = math.ceil(
                    _minibatch_end_idx - _minibatch_start_idx / self._gradient_batch_size)
                for accumulated_batch in range(gradient_accumulation_steps):
                    _start_idx = _minibatch_start_idx + accumulated_batch * self._gradient_batch_size
                    _stop_idx = _minibatch_start_idx + min(
                        (accumulated_batch + 1) * self._gradient_batch_size, _minibatch_end_idx)

                    _contexts = contexts[_start_idx:_stop_idx]
                    _candidates = candidates[_start_idx:_stop_idx]
                    if len(_contexts) == 0: break
                    if self._gradient_minibatch_size is None:
                        _batch_size = sum(len(_c) for _c in _candidates)
                    else:
                        _batch_size = self._gradient_minibatch_size
                    # Use LLM to compute again action probabilities and value
                    output = self._llm_module(['score', 'value'], contexts=_contexts, candidates=_candidates,
                                              require_grad=True, minibatch_size=_batch_size)
                    scores = torch.stack([_o['score'] for _o in output]).squeeze()
                    probas = torch.distributions.Categorical(logits=scores)
                    values = torch.stack([_o["value"][0] for _o in output]).squeeze()

                    # Compute policy loss
                    entropy = probas.entropy().mean()
                    log_prob = probas.log_prob(current_process_buffer['actions'][_start_idx:_stop_idx]) # Use logprobs from dist as they were normalized
                    ratio = torch.exp(log_prob - current_process_buffer['logprobs'][_start_idx:_stop_idx])
                    # assert not (i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)))
                    if i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)):
                        logging.warning("PPO ratio != 1 !!")

                    clip_adv = torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]) * current_process_buffer['advantages'][_start_idx:_stop_idx]
                    policy_loss = -(torch.min(ratio * current_process_buffer['advantages'][_start_idx:_stop_idx], clip_adv)).mean()
                    epochs_losses["policy"].append(policy_loss.detach().cpu().item())

                    # Compute value loss
                    unclipped_value_error = ((values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    clipped_values = current_process_buffer['values'][_start_idx:_stop_idx] + \
                                     torch.clamp(values - current_process_buffer['values'][_start_idx:_stop_idx],
                                                 -kwargs["clip_eps"], kwargs["clip_eps"])
                    clipped_value_error = ((clipped_values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    value_loss = torch.max(unclipped_value_error, clipped_value_error).mean()
                    epochs_losses["value"].append(value_loss.detach().cpu().item())

                    # Compute final loss
                    loss = policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss
                    loss = loss / gradient_accumulation_steps
                    epochs_losses["loss"].append(loss.detach().cpu().item())

                    # Backward
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self._iterator_trainable_params, kwargs["max_grad_norm"])
                self.optimizer.step()

        if kwargs["save_after_update"] and accelerator.process_index == 1:
            print("Saving model...")
            model_state_dict = OrderedDict({
                    k: v for k, v in self._iterator_named_trainable_params()
                })
            torch.save(model_state_dict, kwargs["output_dir"] + "/model.checkpoint")
            torch.save(self.optimizer.state_dict(), kwargs["output_dir"] + "/optimizer.checkpoint")
            print("Model saved")

        return {'loss': np.mean(epochs_losses["loss"]), 'value_loss': np.mean(epochs_losses["value"]),
                'policy_loss': np.mean(epochs_losses["policy"])}


def reset_history():
    return {
        "ep_len": [],
        "ep_ret": [],
        "goal": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "possible_actions": [],
        "actions": [],
        "prompts": [],
    }

@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # Random seed
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    
    # Instantiate environment
    name_env = config_args.rl_script_args.name_environment
    envs = EnvEnum[name_env].value(config_args.rl_script_args)

    # Create LLM agent
    lm_server = Caller(config_args.lamorel_args,
                       custom_updater=PPOUpdater(config_args.lamorel_args.llm_args.model_type,
                                                 config_args.rl_script_args.minibatch_size,
                                                 config_args.rl_script_args.gradient_batch_size,
                                                 config_args.rl_script_args.quantized_optimizer),
                       custom_model_initializer=SequentialInitializer([
                           PeftInitializer(config_args.lamorel_args.llm_args.model_type,
                                           config_args.lamorel_args.llm_args.model_path,
                                           config_args.rl_script_args.use_lora,
                                           config_args.lamorel_args.llm_args.load_in_4bit,
                                           config_args.rl_script_args.lora_r,
                                           config_args.rl_script_args.lora_alpha,
                                           config_args.lamorel_args.llm_args.pre_encode_inputs),
                           WeightsLoaderInitializer(None)
                       ]),
                       custom_module_functions={
                            'score': LogScoringModuleFn(config_args.lamorel_args.llm_args.model_type,
                                                        config_args.lamorel_args.llm_args.pre_encode_inputs),
                            'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type,
                                                       config_args.lamorel_args.llm_args.pre_encode_inputs)
                        })
    print("start")
    # Set up experience buffer
    buffers = [
        PPOBuffer(config_args.rl_script_args.steps_per_epoch // config_args.rl_script_args.number_envs,
                  config_args.rl_script_args.gamma, config_args.rl_script_args.lam)
        for _ in range(config_args.rl_script_args.number_envs)
    ]
    
    # Prepare for interaction with environment
    (o, infos), ep_ret, ep_len = envs.reset(), \
        [0 for _ in range(config_args.rl_script_args.number_envs)], \
        [0 for _ in range(config_args.rl_script_args.number_envs)]
    
    prompt_number=config_args.lamorel_args.llm_args.prompt_number
    history = reset_history()
    history["goal"].extend([_i["goal"] for _i in infos])
    transitions_buffer = [[] for _ in range(config_args.rl_script_args.number_envs)]
    
    limite=0
    reussis=0
    if not os.path.exists("/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/outputss"):
        os.mkdir("/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/outputss")
    data={}
    for limite in tqdm(range(nombre_episode)):
            d=False
            _tmp={"reward":False,"episodes":{}}

            step=0
            list_actions=[]
            while not d:
                bot =Bot(envs._env.envs[0] )
                action = bot.replan()
                
            
           
                possible_actions = [_i["possible_actions"] for _i in infos]
                prompts = [pick_a_prompt(_buff, _o, _i,list_actions,prompt_number,step) for _buff, _o, _i in zip(transitions_buffer, o, infos)]
                
                output = lm_server.custom_module_fns(['score', 'value'],
                                                 contexts=prompts,
                                                 candidates=possible_actions)
                scores = scores_stacking([_o['score'] for _o in output])
                proba_dist = torch.distributions.Categorical(logits=scores)
                values = torch.stack([_o["value"][0] for _o in output])
                sampled_actions = proba_dist.sample()
                log_probs = proba_dist.log_prob(sampled_actions)
                actions_id = sampled_actions.cpu().numpy()
                actions_command = []
                for j in range(len(actions_id)):
                    command = possible_actions[j][int(actions_id[j])]
                    actions_command.append(command)
                    transitions_buffer[j].append({"obs": o[j], "act": command})
                    transitions_buffer[j] = transitions_buffer[j][-config_args.rl_script_args.transitions_buffer_len:]
                o, r, d, infos = envs.step(actions_id=actions_id, actions_command=actions_command)
                list_actions.append(actions_command[0])
                
                step+=1
                _tmp["episodes"][str(step)]={
                    "text":prompts[0],
                    "action":actions_command[0],
                    "bot":str(action)
                }
            
            if r[0]!=0:# si l'episodes est terminé ou bien on a atteint la limite de l'episode
                
                
                reussis+=1
                print(r,reussis)
                _tmp["reward"]=True
            data[str(limite)]=_tmp
            
    json_object = json.dumps(data, indent=4)
 
# Writing to sample.json
    with open("/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Zero_shot/outputss/template"+str(prompt_number)+".json", "w") as outfile:
        outfile.write(json_object)
    
    
    
# Writing to sample.json

    lm_server.close()                
                
            
            

    
if __name__ == '__main__':
    main()