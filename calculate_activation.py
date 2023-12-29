import json
import os
import time 
from tqdm import tqdm
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.float16, load_in_4bit= True,  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,device_map='auto')
    print(model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts, task):
    batch_size = 8
    token_top1_expert = torch.zeros(model.config.num_hidden_layers, model.config.vocab_size, model.config.num_local_experts,dtype=torch.long)
    token_top2_expert = torch.zeros(model.config.num_hidden_layers, model.config.vocab_size, model.config.num_local_experts,dtype=torch.long)
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        input_ids = encode_inputs.input_ids.view(-1).cpu()
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
    if not os.path.exists(os.path.join('layer',task)):
        os.mkdir(os.path.join('layer',task))
    torch.save(token_top1_expert, os.path.join('layer',task,'token_top1_expert.pt'))
    torch.save(token_top2_expert, os.path.join('layer',task,'token_top2_expert.pt'))

def main(ckpt_dir: str):
    
    model, tokenizer = load(ckpt_dir)
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1]-1]
            records.append({'prompt':prompt, 'answer':label})

        batch_infer(model, tokenizer, [record['prompt'] for record in records], task)
        print(task)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ntrain', type=int, default=5)
    args = parser.parse_args()
    
    main(args.ckpt_dir)

