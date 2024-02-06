import json
from lamorel.server.llms import HF_LLM
from omegaconf import OmegaConf
from lamorel.server.llms import HF_LLM
import torch
import umap
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from generate_prompt import *
from tqdm import tqdm
reducer = umap.UMAP()
args=OmegaConf.create({'model_type': 'seq2seq', 'model_path': 'google/flan-t5-large', 'pretrained': True, 'minibatch_size': 192, 'pre_encode_inputs': True, 'load_in_4bit': False, 'parallelism': {'use_gpu': True, 'model_parallelism_size': 1, 'synchronize_gpus_after_scoring': False, 'empty_cuda_cache_after_scoring': False}})
model_original=HF_LLM(args,[0],False).cuda().eval()
model_ft=HF_LLM(args,[0],False).cuda().eval()
loaded_ddp_dict = torch.load("/home/aissi/gotoall/model.checkpoint")
hf_llm_module_dict = {_k.replace('module.', ''): _v for _k, _v in loaded_ddp_dict.items()}
model_ft.load_state_dict(hf_llm_module_dict,strict=False)
p0=json.load(open("/home/aissi/large 0/0.json"))
texts=[]
for key in p0:
    if p0[key]["rewarded"]:
        for steps in list(p0[key]["episode"])[2:]:
            texts.append(p0[key]["episode"][steps]["prompt"][0])
            if len(texts)>=1000:
                break
    if len(texts)==1000:
                break
file=open("/home/aissi/Analysing-the-impact-of-RL-based-grounding-on-LLMs/Evaluations/Embedding_analysis/states.txt","w")
for t in texts:
      file.write(t+"\nS")
print("nombre d'etats calculer:",len(texts))
mean_embeddings=np.zeros((6*len(texts),1024))
first_embeddings=np.zeros((6*len(texts),1024))
prompts=[0,prompt_template1,prompt_template2,prompt_template3,prompt_template4,prompt_template6]

for i in tqdm(range(0,6*len(texts),1000)):
      for j in tqdm( range(0,len(texts),1)):
        p=int(i/1000)
        if prompts[p]==0:
             text=texts[j]
        else:
             text=prompts[p](texts[j])
        tokens=model_original._LLM_tokenizer.encode(text,return_tensors="pt")
        first_embeddings[i+j,:]=model_ft._LLM_model.encoder(tokens.cuda()).last_hidden_state[0,0,:].cpu().detach().numpy()
        mean_embeddings[i+j,:]=model_ft._LLM_model.encoder(tokens.cuda()).last_hidden_state[0,:,:].mean(dim=0).cpu().detach().numpy()
np.save("ftfirst_embeddings.npy",first_embeddings)
np.save("ftmean_embeddings.npy",mean_embeddings)
reducer=umap.UMAP()
first_embeddings2D=reducer.fit_transform(first_embeddings)
reducermean=umap.UMAP()
mean_embeddings2D=reducermean.fit_transform(mean_embeddings)

np.save("ftfirst_embeddings2D.npy",first_embeddings2D)
np.save("ftmean_embeddings2d.npy",mean_embeddings2D)
             
      
