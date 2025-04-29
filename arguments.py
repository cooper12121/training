# -*- encoding: utf-8 -*-
'''
@File    :   arguments.py
@Time    :   2024/12/19 
@Author  :   tensorgao 
@Version :   1.0
@Contact :   gaoqiang_mx@163.com
@Desc    :   None
'''

from typing import Dict, Optional,Literal
from dataclasses import dataclass, field
import transformers


@dataclass
class ModelArguments:
    lora_trainable : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default="embed_tokens,lm_head")
    use_lora : Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-moe-16b")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."}),
    eval_path: str = field(default=None, metadata={"help": "Path to the eval data."}),
    eval_output_path: str = field(default=None, metadata={"help": "Path to the eval data."}),


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    ),
    do_train: Optional[bool] = field(default=True),
    do_eval: Optional[bool] = field(default=True),
    # save_safetensors: Optional[bool] = field(
    #         default=False,
    #         metadata={
    #             "help": "Use safetensors saving and loading for state dicts instead of default torch.load and torch.save."
    #         },
    # )
    padding_side:Optional[str] = field(default="right"),
    stage: Optional[Literal['pretraining','sft','dpo','ppo']] = field(default="sft"),



# @dataclass
# class CustomTrainingArguments(transformers.TrainingArguments):
#     # experiment setups
#     reward_domain: str = field(
#         default="general", 
#         metadata={"help": "the domain for reward model training."}
#     )
#     # tokenizer params
#     padding_side: str = field(
#         default="right",
#         metadata={"help": "the direction for tokenizer to add padding tokens."}
#     )

#     truncation_side: str = field(
#         default="right",
#         metadata={"help": "the direction for tokenizer to truncate tokens."}
#     )

#     # model params
#     # model_type: str = field(
#     #     default="llama",
#     #     metadata={"help": "the base model type for reward model, selected from [llama, bert]."}
#     # )

#     pooling_type: str = field(
#         default="average",
#         metadata={"help": "the pooling method for reward model, selected from [average, max, last]."}
#     )

#     model_name_or_path: str = field(
#         default="llama-7b-hf", 
#         metadata={"help": "the path to load pretrained model."}
#     )

#     tokenizer_path: str = field(
#         default="llama-7b-hf", 
#         metadata={"help": "the path to load pretrained tokenizer."}
#     )

#     flash_attn: bool = field(
#         default=False,
#         metadata={"help": "whether use flash attention in model."}
#     )


#     # data params
#     data_path: str = field(
#         default="yahma/alpaca-cleaned",
#         metadata={"help": "the path to load data."}
#     )   

#     data_dir: str = field(
#         default="path/to/cleaned_data",
#         metadata={"help": "the directory to load data."}
#     )   
    
#     max_response_num: int = field(
#         default=2, 
#         metadata={"help": "max response number."}
#     )


#     train_data_path: List[str] = field(
#         default_factory=lambda: ["/data/to/train/dataset"],
#         metadata={"help": "train datasets paths."}
#     )


#     eval_data_path: List[str] = field(
#         default_factory=lambda: ["/data/to/eval/dataset"],
#         metadata={"help": "evaluation datasets paths."}
#     )


#     data_prefix: str = field(
#         default="yahma/alpaca-cleaned",
#         metadata={"help": "the prefix to load train and test data."}
#     )   


#     text_key: str = field(
#         default="inputs",
#         metadata={"help": "The key name for text inputs."}
#     )
    
#     score_key: str = field(
#         default="scores",
#         metadata={"help": "The key name for scores inputs."}
#     )

#     format_mode: str = field(
#         default="Baichuan",
#         metadata={"help": "the format to process data"}
#     )
        

#     # training hyperparams    
#     debug_mode: bool = field(
#         default=False,
#         metadata={"help": "whether use the debug mode."}
#     )

#     cache_dir: Optional[str] = field(default=None)

#     optim: str = field(default="adamw_torch")

#     max_length: int = field(
#         default=256,
#         metadata={"help": "the max sentence sequence length."}
#     )   

#     batch_size: int = field(
#         default=256,
#         metadata={"help": "the overall training batch size"}
#     )   

#     micro_batch_size: int = field(
#         default=32,
#         metadata={"help": "the batch size on each device, equavilent to `per_gpu_train_batch_size`"}
#     )


#     valid_data_size: int = field(
#         default=0,
#         metadata={"help": "the data size for validation data"}
#     )

#     resume_from_checkpoint: Optional[str] = field(
#         default=None, 
#         metadata={"help":  "either training checkpoint or final adapter"}
#     )

#     # lora hyperparams
#     lora_r: int = field(
#         default=8,
#         metadata={"help": "parameter r for lora."}
#     )

#     lora_alpha: int = field(
#         default=16,
#         metadata={"help": "parameter alpha for lora."}
#     )

#     lora_dropout: float = field(
#         default=0.05,
#         metadata={"help": "dropout rate for lora."}
#     )

#     lora_target_modules: List[str] = field(
#         default_factory=lambda: ["q_proj","v_proj"],
#         metadata={"help": "target modules for lora optimization."}
#     )


    # # llm hyperparams
    # train_on_inputs: bool = True  # if False, masks out inputs in loss
    # add_eos_token: bool = False
    # group_by_length: bool = False  # faster, but produces an odd training loss curve
    # # wandb params
    # wandb_project: str = ""
    # wandb_run_name: str = ""
    # wandb_watch: str = ""  # options: false | gradients | all
    # wandb_log_model: str = ""  # options: false | true
    # prompt_template_name: str = "alpaca"  # The prompt template to use, will default to alpaca.


    # def __post_init__(self):
    #     self.per_device_train_batch_size = self.micro_batch_size

    #     self.evaluation_strategy="steps" if self.valid_data_size > 0 else "no"
    #     # save_strategy="steps", # default
    #     #self.eval_steps=200 if self.valid_data_size > 0 else None
    #     # save_steps=200,
    #     # save_total_limit=3,
    #     self.load_best_model_at_end=True if self.valid_data_size > 0 else False

    #     super().__post_init__()

        # ddp_find_unused_parameters=False if ddp else None,
        # group_by_length=group_by_length,
        # report_to="wandb" if use_wandb else None,
        # run_name=wandb_run_name if use_wandb else None,
    


