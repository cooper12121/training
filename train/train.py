
import copy
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import logging
import os,sys

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset
import datasets
import numpy as np
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import Dataset

from utils import print_rank_0,logger
from arguments import ModelArguments, DataArguments, TrainingArguments
from data_utils import SupervisedDataset, DataCollatorForSupervisedDataset,DataCollatorForSupervisedDataset_inside
# from load_model import get_last_checkpoint, safe_save_model_for_hf_trainer, build_model,SavePeftModelCallback

sys.path.append("/apdcephfs_qy3/share_301372554/share_info/qianggao/")


# -*- encoding: utf-8 -*-
'''
@File    :   load_model.py
@Time    :   2024/12/19 
@Author  :   tensorgao 
@Version :   1.0
@Contact :   gaoqiang_mx@163.com
@Desc    :   None
'''


import torch
import os
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft.tuners.lora import LoraLayer


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)
        

def print_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )



def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def build_model(model_args, training_args, checkpoint_dir):
    if not model_args.use_lora: assert model_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if training_args.bf16 else torch.float16)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=model_args.bits == 4,
        #     load_in_8bit=model_args.bits == 8,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=compute_dtype,
        #     bnb_4bit_use_double_quant=model_args.double_quant,
        #     bnb_4bit_quant_type=model_args.quant_type,
        # ) if model_args.use_lora else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        use_cache=False,
        # output_router_logits=True,##注意添加：计算router的loss
    )

    if compute_dtype == torch.float16 and model_args.bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info('='*80)
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('='*80)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32

    
    # Tokenizer
    
    if model_args.use_lora and model_args.bits < 16:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # if model_args.use_lora:
    #     if checkpoint_dir is not None:
    #         logger.info(f"Loading adapters from {checkpoint_dir}.")
    #         # os.path.join(checkpoint_dir, 'adapter_model')
    #         model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
    #     else:
    #         logger.info(f'Init LoRA modules...')
    #         target_modules = model_args.lora_trainable.split(',')
    #         modules_to_save = model_args.modules_to_save
    #         if modules_to_save is not None:
    #             modules_to_save = modules_to_save.split(',')
    #         lora_rank = model_args.lora_rank
    #         lora_dropout = model_args.lora_dropout
    #         lora_alpha = model_args.lora_alpha
    #         peft_config = LoraConfig(
    #             task_type=TaskType.CAUSAL_LM,
    #             target_modules=target_modules,
    #             inference_mode=False,
    #             r=lora_rank, lora_alpha=lora_alpha,
    #             lora_dropout=lora_dropout,
    #             modules_to_save=modules_to_save)
            

    #         model = get_peft_model(model, peft_config)

    try:
        model.print_trainable_parameters()
    except Exception:
        print_trainable_parameters(model)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    # from deepspeed.utils import set_z3_leaf_modules
    # # from modeling_file.deepseek_moe.modeling_deepseek import DeepseekMoE
    # # from  modeling_file.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
    # # from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    

    # #  Set z3 flag to make sparse MoE layer compatible with Zero3,
    # # following https://github.com/microsoft/DeepSpeed/pull/5008
    # set_z3_leaf_modules(model, [LlamaSparseMoeBlock])
    
    return model


def load_data(file_path:str):
    def jsonline_load(f, mode="r"):
        """Load a .jsonl file into a dictionary."""
        f = _make_r_io_base(f, mode)
        json_objects_list = [json.loads(line) for line in f]
        return json_objects_list

    if ".jsonl" in file_path:
        data_list = jsonline_load(file_path)
    elif ".json" in file_path:
        data_list = json.load(open(file_path,'r'))
    return data_list

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank == 0:
        logger.info('='*100)
        logger.info(training_args)
    
   
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side=training_args.padding_side,
        use_fast=False,
        trust_remote_code=True
    )

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    
    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
         tokenizer.pad_token_id = tokenizer.eos_token_id

    if training_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(model_args.model_name_or_path))
    
   
    # if training_args.local_rank==0:
    torch.distributed.barrier()
   
    
    resume_from_checkpoint_dir = get_last_checkpoint(training_args.output_dir)
    
    print_rank_0(f"\n\n****resulme from :{resume_from_checkpoint_dir}****\n\n")
    model = build_model(model_args, training_args, resume_from_checkpoint_dir)

    additional_special_tokens ={
    "additional_special_tokens": ["<Retrieval>","<No Retrieval>","<Relevant>","<Irrelevant>"],
    }
    num_added_toks = tokenizer.add_special_tokens(additional_special_tokens)
    print_rank_0(f"Added {num_added_toks} special tokens")
    model.resize_token_embeddings(len(tokenizer))
   
    
    if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    if training_args.stage=='sft':
        print_rank_0(f"stage: sft,start to load dataset")

        train_dataset = SupervisedDataset(data_args.data_path,tokenizer=tokenizer,format_mode="llama2")
        # how to shuffle the dataset
        
        train_dataset = datasets.Dataset.from_list(list(train_dataset)) # Convert to list if it's an iterator
        train_dataset.shuffle()
        
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        # train_dataset = datasets.Dataset.from_list(load_data(data_args.data_path))
        # data_collator = DataCollatorForSupervisedDataset_inside(tokenizer=tokenizer)
        # training_args.remove_unused_columns=False

        print_rank_0(f"Training dataset samples:{len(train_dataset)}")

        data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics=None,**data_module)
        if model_args.use_lora:
            trainer.add_callback(SavePeftModelCallback)

    elif training_args.stage=='dpo':
        from trl import DPOConfig, DPOTrainer
        from data_utils import RMDataset

        # training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10)
        training_args=DPOConfig.from_training_args(training_args)
        

        # dpo dataset的传入还是保持["prompt","chosen","rejected"]的标准格式构建即可
        # notice that the type of train_dataset if datasets.Dataset
        train_dataset = RMDataset(data_path=data_args)
        trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
     

    
    if training_args.do_train:
        logger.info("*** Training ***")
        trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
        trainer.save_state()
        # if not model_args.use_lora:
        #     safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        trainer.save_model(output_dir=training_args.output_dir)
        # tokenizer.save_pretrained(save_directory=training_args.output_dir)


if __name__ == "__main__":
    train()
