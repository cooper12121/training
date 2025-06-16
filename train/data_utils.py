# -*- encoding: utf-8 -*-
'''
@File    :   arguments.py
@Time    :   2024/12/19 
@Author  :   tensorgao 
@Version :   1.0
@Contact :   gaoqiang_mx@163.com
@Desc    :   None
'''

import torch.distributed
import transformers
import json
import numpy as np
import torch
import pandas
import random

from torch.utils.data import Dataset
import copy 
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence,Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


from utils import print_rank_0,logger



IGNORE_INDEX = -100
def _make_r_io_base(f, mode: str):
    import io
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jsonline_load(f, mode="r"):
    """Load a .jsonl file into a dictionary."""
    f = _make_r_io_base(f, mode)
    json_objects_list = [json.loads(line) for line in f]
    return json_objects_list

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            # return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding='longest'
        )
        for text in strings
    ]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        len(tokenized.input_ids) for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, format_mode:str="qwen2",data_type:str="train") -> None:
        super(SupervisedDataset).__init__()

        rank = torch.distributed.get_rank()

        if ".xlsx" in data_path:
            if data_type=="train":
                data_list = pandas.read_excel(data_path).to_dict(orient="records")
            elif data_type=="test":
                data_list = pandas.read_excel(data_path).to_dict(orient="records")
        elif ".jsonl" in data_path:
            data_list = jsonline_load(data_path)
        elif ".json" in data_path:
            data_list = json.load(open(data_path,'r'))
        if rank==0:
            print_rank_0(f"len of data_list:{len(data_list)}")
        if data_type=="train":
            random.shuffle(data_list)
            
        if format_mode=="qwen2":
            """"""
            PROMPT_FORMAT_SYSTEM_Qwen = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
            PROMPT_FORMAT_MULTI_Qwen = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>\n"
            PROMPT_FORMAT_SINGLE_Qwen = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

           
            sources = []
            targets = []

            for data in tqdm(data_list,desc="data_list"):
                sources.append(PROMPT_FORMAT_SYSTEM_Qwen+PROMPT_FORMAT_SINGLE_Qwen.format(data['instruction']))
                targets.append(data['output']+tokenizer.eos_token)
            print_rank_0(f"{sources[0]}{targets[0]}\n{type(sources)},{len(sources)}")
            data_dict = preprocess(sources, targets, tokenizer)
            
        elif format_mode=="llama2":
            PROMPT_FORMAT_SYSTEM_llama2 = '<<SYS>>\n' +"You are a helpful assistant"+ '\n<</SYS>>\n\n'
            PROMPT_FORMAT_SINGLE_llama2 = "<s>[INST] {instruction} [/INST]"
            sources = []
            targets = []

            for data in tqdm(data_list,desc="data_list"):
                message = [
                    # {"role":"system","content":"you are a helpful assistant"},
                    {"role":"user","content":data['instruction']}
                ]
                # sources.append(PROMPT_FORMAT_SINGLE_llama2.format(instruction=data['instruction']))
                sources.append(tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=False))
                targets.append(' '+data['output']+' '+tokenizer.eos_token)
            print_rank_0(f"{sources[0]}{targets[0]}\n{type(sources)},{len(sources)}")
            data_dict = preprocess(sources, targets, tokenizer)
            
            
        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']
        
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i]) 
    
    def shuffle(self, seed=None):
        """In-place shuffle of the dataset, keeping input_ids and labels in sync."""
        if seed is not None:
            random.seed(seed)
        indices = list(range(len(self.input_ids)))
        random.shuffle(indices)
        self.input_ids = [self.input_ids[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        #  if left padding is needed:
        # input_ids = self.left_pad_sequence(input_ids, self.tokenizer.pad_token_id)
        # labels = self.left_pad_sequence(labels, IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
    def left_pad_sequence(self,sequences: Sequence[torch.Tensor], padding_value: int) -> torch.Tensor:
        """ 
        Pad a list of sequences on the left.(batch-level)
        """
        max_len = max(len(seq) for seq in sequences)
        # Pad sequences on the left manually
        padded_sequences = [torch.cat([torch.full((max_len - len(seq),),padding_value, dtype=seq.dtype), seq]) for seq in sequences]

        # Stack the padded sequences into a tensor
        padded_tensor = torch.stack(padded_sequences)

        return padded_tensor


@dataclass
class DataCollatorForSupervisedDataset_inside(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, data_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sources = []
        targets = []
        for data in tqdm(data_list,desc="data_list"):
            print(data,data.keys())
            message = [
                # {"role":"system","content":"you are a helpful assistant"},
                {"role":"user","content":data['instruction']}
            ]
            sources.append(self.tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True))
            targets.append(' '+data['output']+' '+self.tokenizer.eos_token)
        # print_rank_0(f"{sources[0]}{targets[0]}\n{type(sources)},{len(sources)}")
        data_dict = preprocess(sources, targets, self.tokenizer)


        # input_ids, labels = tuple([instance[key] for instance in data_dict] for key in ("input_ids", "labels"))
        input_ids = [data_dict['input_ids'][i]for i in range(len(data_dict['input_ids']))]
        input_ids = [data_dict['labels'][i]for i in range(len(data_dict['labels']))]

        input_ids = [torch.tensor(x) for x in input_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )




        # input_ids = []
        # attention_mask = []
        # prompts_input_ids = []
        # prompt_attention_mask = []
        # labels = []

        # for example in examples:
        #     formatted_prompt = example.get(self.prompt_key, None)
        #     if formatted_prompt is None:
        #         prompt = example[self.messages_key][:-1]
        #         formatted_prompt = self.tokenizer.apply_chat_template(
        #             prompt, tokenize=False, add_generation_prompt=True
        #         )

        #     if "input_ids" not in example:
        #         message = example[self.messages_key]
        #         formatted_message = self.tokenizer.apply_chat_template(
        #             message, tokenize=False, add_generation_prompt=False
        #         )
        #         tokenized_message = self.tokenizer(
        #             formatted_message,
        #             truncation=True,
        #             max_length=self.max_length,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=False,
        #         )
        #         input_ids.append(tokenized_message["input_ids"])
        #         attention_mask.append(tokenized_message["attention_mask"])
        #     else:
        #         input_ids.append(example["input_ids"])
        #         attention_mask.append(example["attention_mask"])

        #     tokenized_prompt = self.tokenizer(
        #         formatted_prompt,
        #         truncation=True,
        #         max_length=len(input_ids[-1]),
        #         padding=False,
        #         return_tensors=None,
        #         add_special_tokens=False,
        #     )

        #     prompts_input_ids.append(tokenized_prompt["input_ids"])
        #     prompt_attention_mask.append(tokenized_prompt["attention_mask"])

        #     # Create the labels that will have all but the completion tokens of the example["input_ids"] set to ignore_index
        #     label = [self.ignore_index] * len(input_ids[-1])
        #     completion_start_idx = len(tokenized_prompt["input_ids"])
        #     label[completion_start_idx:] = input_ids[-1][completion_start_idx:]
        #     labels.append(label)

        # # convert to list of tensors and pad
        # input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        # attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in attention_mask]
        # labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        # input_ids = pad(input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        # attention_mask = pad(attention_mask, padding_side="left", padding_value=0)
        # labels = pad(labels, padding_side="left", padding_value=self.ignore_index)

        # prompts_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in prompts_input_ids]
        # prompt_attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in prompt_attention_mask]
        # prompts_input_ids = pad(prompts_input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        # prompt_attention_mask = pad(prompt_attention_mask, padding_side="left", padding_value=0)

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": labels,
        #     "prompts": prompts_input_ids,
        #     "prompt_attention_mask": prompt_attention_mask,
        # }
def RMDataset(data_path:str):
    """ 
    dataset for the RM task. such as reward model, dpo etc.
    # notice that the type of this dataset if datasets.Dataset
    """
        # here we need to load the data from the data_path and transform it into the dataset format

    def return_prompt_and_responses(self,samples):
        """ directly return the final format in DPOTrainer
        {
        "prompt": [{"role": "user", "content": "You are a helpful assistant"}]
        "chosen": [{"role": "assistant", "content": "You are a helpful assistant"}]
        "rejected":[{"role": "assistant", "content": "You are a helpful assistant"}]
        }
        {
        "prompt": [{"role": "user", "content": "You are a helpful assistant"}]
        "chosen": [{"role": "assistant", "content": "You are a helpful assistant"}]
        "rejected":[{"role": "assistant", "content": "You are a helpful assistant"}]
        }
        
        """
        prompts = []
        chosens = []
        rejecteds = []
        
        for prompt, chosen, rejected in zip(samples["prompt"], samples["chosen"], samples["rejected"]):
                # 假设每个样本是一个字典，包含 'prompt', 'chosen', 'rejected' 键
                prompts.append([{"role": "user", "content": prompt}])
                chosens.append([{"role": "assistant", "content": chosen}])
                rejecteds.append([{"role": "assistant", "content": rejected}])
        
        return {
                "prompt": prompts,
                "chosen": chosens,
                "rejected": rejecteds
        }
    
    import datasets
    data_dict = jsonline_load(data_path)
    data_dict = datasets.Dataset.from_list(data_dict)
    dataset = data_dict.map(return_prompt_and_responses,batched=True,batch_size=1000,remove_columns=data_dict.column_names)
    return dataset
