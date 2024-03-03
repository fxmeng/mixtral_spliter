import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def infer_router_logits(token_top1_expert, token_top2_expert, model, tokenizer, inputs):
    for prompt in tqdm(inputs):
        encode_inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = encode_inputs.input_ids.view(-1)
        with torch.no_grad():
            outputs = model(**encode_inputs, output_router_logits=True)
        for l in range(model.config.num_hidden_layers):
            top1=torch.argsort(outputs['router_logits'][l],dim=-1)[:,-1].cpu()
            top2=torch.argsort(outputs['router_logits'][l],dim=-1)[:,-2].cpu()
            for e in range(model.config.num_local_experts):
                tokens1 = input_ids[torch.where(top1==e)]
                token_top1_expert[l,:,e].index_add_(0, tokens1, torch.ones_like(tokens1, dtype=token_top1_expert.dtype))
                tokens2 = input_ids[torch.where(top2==e)]
                token_top2_expert[l,:,e].index_add_(0, tokens2, torch.ones_like(tokens2, dtype=token_top2_expert.dtype))
    return token_top1_expert, token_top2_expert


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--text_field', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='layer',required=False)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_dir, torch_dtype=torch.float16, device_map='auto')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    print(model)
    dataset = load_dataset(args.data_dir)
    print(dataset)
    for task in dataset.keys():
        if not os.path.exists(os.path.join(args.out_dir,task)):
            os.mkdir(os.path.join(args.out_dir,task))
        print(task)
        token_top1_expert = torch.zeros(model.config.num_hidden_layers, model.config.vocab_size, model.config.num_local_experts,dtype=torch.long)
        token_top2_expert = torch.zeros(model.config.num_hidden_layers, model.config.vocab_size, model.config.num_local_experts,dtype=torch.long)
        if not os.path.exists(os.path.join(args.out_dir,task)):
            os.mkdir(os.path.join(args.out_dir,task))
        token_top1_expert, token_top2_expert = infer_router_logits(token_top1_expert, token_top2_expert, model, tokenizer, dataset[task][args.text_field])
        torch.save(token_top1_expert, os.path.join(args.out_dir,task,'token_top1_expert.pt'))
        torch.save(token_top2_expert, os.path.join(args.out_dir,task,'token_top2_expert.pt'))
