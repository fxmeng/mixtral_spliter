import fire
import os
import json
import shutil
from safetensors import safe_open
from safetensors.torch import save_file

def select_experts(experts_ids=(0,2), source_dir = "/home/mfx/code/huggingface/Mixtral-8x7B-Instruct-v0.1/", output_dir = "/home/mfx/code/huggingface/Mixtral-0_1", num_experts_per_tok=2):
    print("experts_ids: ", experts_ids)
    print("source_dir: ", source_dir)
    print("output_dir: ", output_dir)
    if isinstance(experts_ids, int):
        experts_ids = [experts_ids]
    elif isinstance(experts_ids, tuple):
        experts_ids = list(experts_ids)
    else:
        raise
    
    assert len(set(experts_ids))==len(experts_ids)
    with open(os.path.join(source_dir, 'config.json'))as f:
        config = json.load(f)
        config['num_experts_per_tok']=num_experts_per_tok
        config['num_local_experts']=len(experts_ids)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir,'config.json'), 'w') as file:
            json.dump(config, file)
    shutil.copy(os.path.join(source_dir, "special_tokens_map.json"), os.path.join(output_dir, "special_tokens_map.json"))
    shutil.copy(os.path.join(source_dir, "tokenizer_config.json"), os.path.join(output_dir, "tokenizer_config.json"))
    shutil.copy(os.path.join(source_dir, "tokenizer.json"), os.path.join(output_dir, "tokenizer.json"))
    shutil.copy(os.path.join(source_dir, "tokenizer.model"), os.path.join(output_dir, "tokenizer.model"))
    file_list = os.listdir(source_dir)
    
    weight_map = {}
    for file in file_list:
        if file.endswith("safetensors"):
            tensors = {}
            with safe_open(os.path.join(source_dir, file), framework="pt", device='cpu') as f:
                for k in f.keys():
                    if 'gate' in k:
                        current_tensors = f.get_tensor(k)
                        current_tensors = current_tensors[experts_ids]
                        tensors[k] = current_tensors
                        weight_map[k]=file
                    elif 'experts' in k:
                        for i,ids in enumerate(experts_ids):
                            if int(k.split('.')[5])==ids:
                                current_tensors = f.get_tensor(k)
                                k = k.replace("experts."+k.split('.')[5], "experts."+str(i))
                                tensors[k] = current_tensors
                                weight_map[k]=file
                    else:
                        current_tensors = f.get_tensor(k)
                        tensors[k] = current_tensors
                        weight_map[k]=file
            save_file(tensors, os.path.join(output_dir, file), metadata={"format":"pt"})
    with open(os.path.join(output_dir,"model.safetensors.index.json"), 'w') as f:
        json.dump({
            "metadata": {
                "total_size": 93405585408
            },
            "weight_map": weight_map
            }, f)
        
if __name__ == '__main__':
  fire.Fire(select_experts)
