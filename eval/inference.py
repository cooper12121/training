import argparse
import json
import logging
import os
import random
from datetime import datetime

import torch
# from accelerate import Accelerator
from tqdm import tqdm
import deepspeed
from torch.utils.data import Dataset, DataLoader,DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
    LlamaTokenizerFast,
)

from arguments import CustomTrainingArguments
# from evaluation import eval_question_answering
import transformers

random.seed(42)
logger = logging.getLogger(__name__)

# 自定义数据集类
class SupervisedDataset(Dataset):
    def __init__(self, file_name, local_rank, world_size):
        self.file_name = file_name
        self.local_rank = local_rank
        self.world_size = world_size
        self.data_list = json.load(open(self.file_name,'r'))
    def __len__(self):
        # with open(self.file_name, 'r') as f:
        #     return sum(1 for _ in f) // self.world_size
        return len(self.data_list) // self.world_size

    def __getitem__(self, index):
        global_index = index * self.world_size + self.local_rank
        # with open(self.file_name, 'r', encoding='utf-8') as f:
        #     for i, line in enumerate(f):
        #         if i == global_index:
        #             return json.loads(line)
        return self.data_list[global_index]


def format_input_prompt(instruction, tokenizer, system=None):
    """
    This method must be consistent with encode_with_messages_format in train.py
    """
    prompt = ""

    if system is not None:
        message  = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
    else:
        message = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return prompt

def main():
    # parser = argparse.ArgumentParser()
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    parser.add_argument("--model", type=str, default=None)
    # parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--precision", 
        type=str, 
        default="bf16", 
        choices=["fp32", "fp16", "bf16"],
    )
    
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1536,
        help="Max sequence length for the instruction.",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=2048,
        help="Max sequence length for generating the response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling, 0.0 means greedy decoding",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="Top-p for sampling, 0.0 means greedy decoding",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for sampling, 1.0 means no penalty",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="if specified, use a subset of alpaca_eval",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help='If specified, only use the first "num_examples" examples in the dataset.',
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="If specified, overwrite the original output file (if exists).",
    )
    parser.add_argument(
        "--continue_output",
        action="store_true",
        help="If specified, continue writing to the original output file (if exists).",
    )
    # parser.add_argument(
    #     "--local_rank",
    #     type=int,
    #     default=-1,
    #     help="Local rank for distributed training. This will be automatically set by the launcher script.",
    # )

    args = parser.parse_args()
    # print(args)
    deepspeed.init_distributed()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("loading data and model...")
    # load some data
    # eval_data = json.load(open(args.data, "r", encoding="utf-8"))
    # eval_data = eval_data[ :len(eval_data) - len(eval_data) % 8]

    # select the specified subset
    # if args.subset is not None:
    #     eval_data = [x for x in eval_data if x["dataset"] == args.subset]

    # if args.num_examples is not None:
    #     eval_data = eval_data[: args.num_examples]

    # logger.info(f"Total evaluation data: {len(eval_data)}")

    '''
    # prev_data = None
    if os.path.exists(args.output_path) or all([os.path.exists(args.output_path+f'.{i}') for i in range(8)]):
        # if args.continue_output:
        #     prev_data = json.load(open(args.output_path, "r", encoding="utf-8"))
        #     prev_data_ids = {x["idx"] for x in prev_data}
        #     logger.warning(
        #         f"Continue writing to {args.output_path}, which already has {len(prev_data)} examples..."
        #     )
        #     eval_data = [x for x in eval_data if x["idx"] not in prev_data_ids]
        # else:
        logger.warning("File %s already exists, exiting...", args.output_path)
        return
    '''
    
    my_outputs = []

    if args.precision == "fp32":
        precision = torch.float32
    elif args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    else:
        raise ValueError("Unknown precision %s", args.precision)

    if "polylm" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=precision, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, legacy=False, use_fast=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=precision
        )


    logger.info("model and data loaded!")
    logger.info("generating...")

    # generation_config = GenerationConfig.from_pretrained(
    #     args.model,
    #     max_length=args.max_output_length,
    #     top_p=args.top_p,
    #     temperature=args.temperature,
    #     do_sample=do_sample,
    #     repetition_penalty=args.repetition_penalty,
    # )
    

    # random.shuffle(eval_data)
    # accelerator = Accelerator()
    model_hidden_size = 4096
    ds_config = {
        "fp16": {
            "enabled": True
        },
        "bf16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto"
        },
        "steps_per_print": 20,
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False
    }
    model_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    local_rank = model_engine.local_rank
    world_size = model_engine.world_size
    model_engine.module.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # add padding token if not already there (for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    logger.warning(
        f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {local_rank}> Start generating..."
    )

    # eval_data_curr_process = eval_data.shard(local_rank) 
    # dataloader = Dataloader(eval_data, batch_size=args.batch_size, shuffle=False)
    dataset = SupervisedDataset(args.data, local_rank, world_size)
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,pin_memory=True,)
    generation_config=GenerationConfig(
                        max_length=args.max_input_length,
                        max_new_tokens=args.max_output_length,
                        temperature=0.2, 
                        top_p=0.85,
                        top_k=40,
                        repetition_penalty=1.05,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    )
    all_outputs = []

    for samples in tqdm(dataloader, desc=f"GPU {local_rank}"):
        input_texts = [
            format_input_prompt(samples["input"][j],
            tokenizer,
            system="你是一个乐于助人的人工智能助手") for j in range(len(samples["id"]))                  
        ]
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=args.max_input_length,
            padding=True,
            truncation=True,
        )
        # input_ids = inputs.input_ids.to(model_engine.device)
        # attention_mask = inputs.attention_mask.to(model_engine.device)
        input_ids = inputs.input_ids.to(f"cuda:{local_rank}")
        attention_mask = inputs.attention_mask.to(f"cuda:{local_rank}")
        try:
            with torch.no_grad():
                outputs = model_engine.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # return_tensors="pt",
                    generation_config=generation_config,
                    use_cache=False
                    
                )
            torch.cuda.empty_cache()
        except Exception as e :
            logging.warning(f"error:{e}")
            continue
        for j in range(len(samples["id"])):
            output = outputs[j]
            output_string = tokenizer.decode(
                output[input_ids.size(1) :], skip_special_tokens=True
            )
            
            my_outputs.append(
                {
                    "idx": samples["id"][j].item(),
                    "generator": f"{args.model}",
                    "input": samples["input"][j],
                    "original_events": samples["original_events"][j],
                    "output": samples["output"][j],
                    "model_output": output_string.strip(),
                }
            )
        
        

    output_path_curr_process = args.output_path + f".{local_rank}"
    args.output_dir = args.output_path.rsplit('/', 1)[0]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    json.dump(
        my_outputs,
        open(output_path_curr_process, "w", encoding="utf8"),
        indent=4,
        ensure_ascii=False,
    )

    logger.warning(
        f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {local_rank}> Finished generation!"
    )

    # accelerator.wait_for_everyone()
    # model_engine.local_barrier()
    '''
    if accelerator.is_main_process:
        # concatenate outputs from all processes
        all_outputs = []
        for i in range(accelerator.num_processes):
            output_path_curr_process = args.output_path + f".{i}"
            all_outputs += json.load(
                open(output_path_curr_process, "r", encoding="utf-8")
            )
            os.remove(output_path_curr_process)

        if prev_data is not None:
            all_outputs += prev_data

        all_outputs = sorted(all_outputs, key=lambda x: x["idx"])
        json.dump(
            all_outputs,
            open(args.output_path, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )
        print(f"Saved {len(all_outputs)} examples to {args.output_path}.")
        logger.info(all_outputs[0])
        
        recall, em, f1_score, avg_lens = eval_question_answering(all_outputs)
        
        with open(args.output_path + '.metrics', "w") as metric_file:
            metric_file.write(
                f"---\nTest Set Results\n---\nRecall: {recall}\nEM: {em}\nF1: {f1_score}\nLens: {avg_lens}\n"
            )
    '''

if __name__ == "__main__":
    
    main()
