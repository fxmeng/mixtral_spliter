import os
import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)


torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
# training and evaluation use the same dataset
dataset = load_dataset("fxmeng/alpaca_in_mixtral_format", split="train")
# load model in 4bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["gate"],
)
config = AutoConfig.from_pretrained('/scratch/nlp/fxmeng/huggingface/Mixtral-8x7B-Instruct-v0.1')
config.use_cache = False
config.gradient_checkpointing = True

model = AutoModelForCausalLM.from_pretrained('/scratch/nlp/fxmeng/huggingface/Mixtral-8x7B-Instruct-v0.1',
                                             config=config,
                                             quantization_config=bnb_config,
                                             trust_remote_code=False,
                                             torch_dtype=torch_dtype,
                                             device_map="auto")

tokenizer = AutoTokenizer.from_pretrained('/scratch/nlp/fxmeng/huggingface/Mixtral-8x7B-Instruct-v0.1',
                                          trust_remote_code=False,
                                          use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['lm_head'],
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, peft_config)
init_state_dict = {n:p.detach().cpu().clone() for n, p in model.named_parameters() if 'gate' in n}

class ExpertsLassoTrainer(SFTTrainer):
    gate_state_dict = {}
    argsort = torch.zeros(32,8,dtype=torch.long)
    def create_optimizer(self):   
        # Put all gate layer into optimizer
        optimizer_grouped_parameters = [{"params": [p for n, p in self.model.named_parameters() if 'gate' in n],
                                         "weight_decay": self.args.weight_decay,
                                         "lr": self.args.learning_rate,},]
        optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        print(self.optimizer)
        # Pruning start from the first layer
        for n,p in self.model.named_parameters():
            if 'model.layers.0.block_sparse_moe.gate.weight' in n:
                p.requires_grad=True
                print(n,p)
            else:
                p.requires_grad=False 
        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        threshold = 0.05
        n_experts = 4
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        for n,p in model.named_parameters():
            if 'gate' in n and p.requires_grad==True:
                experts_lasso = torch.norm(p, p=2, dim=1)
                print(loss, n, experts_lasso, len(self.gate_state_dict))
                if (experts_lasso>threshold).sum()>n_experts:
                    # If hasn't finist pruning current gate layer, add experts lasso loss
                    loss += experts_lasso.sum() 
                else:
                    # If finished pruning current gate layer, record its weights for reordering
                    self.gate_state_dict[n]=p.detach().cpu().clone()
                    argsort = torch.argsort(experts_lasso.detach().cpu().clone(), descending=True)
                    print(n, argsort)
                    # Freeze current gate layer
                    p.requires_grad=False
                    # Restore initial weights of current gate layer
                    p[argsort[:n_experts]]=init_state_dict[n]
                    # delete the less importent experts
                    p[argsort[n_experts:]]=0
                    #current layer is l
                    l = int(n.split('.')[4])
                    # Allow updating l + 1 gate layer
                    for m, q in self.model.named_parameters():
                        if f"model.layers.{l+1}.block_sparse_moe.gate.weight" in m:
                            q.requires_grad=True
                    # Save experts_lasso
                    self.argsort[l]=argsort
                    torch.save(experts_lasso, os.path.join(self.args.output_dir, n+'.bin'))
                            
        # Finish pruning
        if len(self.gate_state_dict)==32:
            torch.save(self.argsort, os.path.join(self.args.output_dir, 'argsort.bin'))
            exit()
        return (loss, outputs) if return_outputs else loss
    
    
# training setting
training_args = TrainingArguments(
    do_train=True,
    output_dir="./checkpoints/alpaca_4experts-Instruct_lasso_wd0_lr0.01_bs1",
    dataloader_drop_last=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    optim="paged_adamw_8bit",
    learning_rate=0.01,
    lr_scheduler_type='constant',
    warmup_steps=50,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    weight_decay=0,
    report_to="wandb",
    save_total_limit=1,
    bf16=torch_dtype==torch.bfloat16,
    fp16=torch_dtype!=torch.bfloat16,
)
print(training_args)
trainer = ExpertsLassoTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
)
trainer.train()
