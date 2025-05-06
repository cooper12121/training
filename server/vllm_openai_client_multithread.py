  

import openai
import requests
import json
import logging
import time
import random
from typing import Generator, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

import os
import pdb
# load_dotenv("./.env")
load_dotenv("/mnt/nlp/gaoqiang/project/training/server/.env")
# 获取变量
model_path = os.getenv("MODEL_PATH")
openai_api_key = os.getenv("API_KEY")
openai_api_base = os.getenv("API_BASE")

log_dir = "/mnt/nlp/gaoqiang/project/training/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
logging.basicConfig(
    filename=f"{log_dir}/vllm_openai_client_multithread_{time_str}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Logging setup complete.")
logging.info(f"** log file: {log_dir}/vllm_openai_client_multithread_{time_str}.log ***")


logging.info(f"model_path: {model_path}")
logging.info(f"openai_api_key: {openai_api_key}")
logging.info(f"openai_api_base: {openai_api_base}")


REQUEST_UNSAFE_STR = 'error_code=content_filter'

class Client(OpenAI):
    def __init__(self,
                 model_name: str = model_path,
                 api_key=openai_api_key,
                 api_base_url: str = openai_api_base,
                 ):
        super().__init__(api_key=api_key, base_url=api_base_url)
        self.model_name = model_name
    def __call__(self, *args, **kwargs):
        return self.complete(*args, **kwargs)

    def _complete(self,
                  messages: list[dict],
                  stream: bool = False,
                  n=1,
                  top_k=50,
                  max_tokens=1500,
                  temperature: float = 0.8,
                  repetition_penalty: float = 1.05,
                  **kwargs) -> requests.Response:
        """
          执行模型调用 
        """
        chat_response = self.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body = {
                "top_k":top_k,
                "n":n,
                "max_tokens":max_tokens,
                "repetition_penalty":repetition_penalty,
                "temperature":temperature,
                # "stop_token_ids":
            },
            stream=stream,
            **kwargs
        )
        return chat_response
    
    def stream_complete(self,
                  chat_response: requests.Response) -> Generator:
        """
            #函数中只要出现了yield，就会变成一个生成器函数
        """ 

        for chunk in chat_response:
            yield chunk.choices[0].delta.content
    
    def non_stream_complete(self,
                  chat_response: requests.Response) -> dict:
        """
            处理非streaming格式下的结果
        """
        return {
                "id": chat_response.id,
                "model": chat_response.model,
                "content": chat_response.choices[0].message.content,
                "finish_reason": chat_response.choices[0].finish_reason,
                "reasoning_content": chat_response.choices[0].message.model_extra['reasoning_content'],                
                "usage": {
                    "total_tokens": chat_response.usage.total_tokens,
                    "prompt_tokens": chat_response.usage.prompt_tokens,
                    "completion_tokens": chat_response.usage.completion_tokens
                }
            }

    def complete(self, messages: list[dict],
                 stream=False,
                 **kwargs) -> Union[requests.Response,dict, Generator]:
        """
            处理api请求的返回结果
                封装非streaming格式下的结果
                处理streaming格式下的结果
        """

        chat_response = self._complete(messages, stream=stream, **kwargs)
        if stream:
            logging.info(f"streaming response")
            return self.stream_complete(chat_response)
        else:
            logging.info(f"non-streaming response")
            return self.non_stream_complete(chat_response)

def process_msg(
            messages: list[dict],
            model_name: str=model_path,
            stream: bool = False,
            max_try: int = 3,
            id: str = None,
            **generate_kwargs) -> tuple[str, list[dict], Union[dict,Generator]]:
        """ 
           parameters:
              messages: list[dict]: 消息列表
              model_name: str: 模型名称
              max_try: int: 最大尝试次数
              id: str: 消息ID
            
        """
        if messages[-1].get("role")=="user" and messages[-1].get("content")==None:
            logging.info(f"*** empty user message: {messages} ***")
            return (id, messages, None)
        
        for i in range(max_try):
            try:
                llm = Client(model_name=model_name)
                res = llm.complete(messages, stream,**generate_kwargs)
                return (id, messages,res)
            except Exception as e:
                # openai.error.InvalidRequestError: content is unsafe, so it was filtered
                # openai.error.RateLimitError: 只有这种情况可以sleep and retry
                _msg = f'Error: {e}, exception type: {type(e)}'
                logging.warning(_msg)
                if any(_ in _msg for _ in ['InvalidRequestError', 'filtered']):
                    logging.error(f'[内容被过滤]: {messages}')
                    return _msg
                time.sleep(random.randint(10, 12))
                continue

def LLM_Caller(
        data_list: list[list[dict]],
        model_name: str=model_path,
        max_workers: int = 1,
        max_try: int = 3,
        text_only: bool = False,
        stream: bool = False,
        multithread: bool = True,
        **generate_kwargs, 
    ):
    
    if multithread:
        # 多线程处理
        logging.info(f"*** multithread: {multithread} ***")
        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            futures = []
            for id, msg in tqdm(enumerate(data_list)):
                future = executor.submit(
                    process_msg,
                    id=id,
                    messages=msg,
                    max_try=max_try,
                    model_name=model_name,
                    stream=stream,
                    **generate_kwargs
                )
                futures.append(future)

            count = 0
            for future in tqdm(as_completed(futures), total=len(futures), ncols=70):
                try:
                    id, messages, resp = future.result()
                    logging.info(f"id: {id}")
                    logging.info(f"prompt:\n{messages}")
                    logging.info(f"response:\n{resp}")

                    logging.info(f"response:\n{resp}")
                    

                    if stream:
                        # TODO: 处理streaming格式下的结果
                        continue

                    if text_only:
                        # 只返回响应文本内容
                        data_list[id].append({
                            "role": "assistant",
                            "content": resp['content'].strip()
                        })
                    else:
                        # 返回完整的响应内容
                        data_list[id].append({
                            "role": "assistant", 
                            "content": resp
                        })
                    yield id,data_list[id]
                except Exception as e:
                    print("error:", e)
                    count += 1
        logging.info(f"total error number:{count}")
    else:
        # 单线程处理
        logging.info(f"*** multithread: {multithread} ***")
        for id, msg in tqdm(enumerate(data_list)):
            try:
                _,message,res = process_msg(
                    id=id,
                    messages=msg,
                    max_try=max_try,
                    model_name=model_name,
                    stream=stream,
                    **generate_kwargs
                )
                data_list[id].append({
                        "role": "assistant", 
                        "content": res
                    })
                yield id,data_list[id]
            except Exception as e:
                print("error:", e)

def load_dataset():
    import os,sys
    from datasets import load_dataset
    path = "google-research-datasets/natural_questions"
    subset = "dev"
    split = "validation"
    dataset_name = path.split('/')[-1]
    dataset = load_dataset(path,subset,split=split,cache_dir=None)
    dataset.to_json(f"/Users/qianggao/project/intern/rag/dataset/{dataset_name}/{subset}/{split}.json")


def data_process(input_path:str=None, output_path:str=None):

    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "1，3，11，13，21...., 下一个数字是多少"},
        ]
    data_list = [messages for _ in range(10)]
    print(f"data_list: {data_list}")
    generate_kwargs = {
        "n":1,
        "max_tokens":1500,
        "repetition_penalty": 1.05,
        "temperature": 0.8
    }
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    result_generator = LLM_Caller(
        data_list,
        # model_name=model_name, # 默认使用.env文件中的MODEL_PATH
        max_workers=10,
        text_only=False,
        multithread=False,
        **generate_kwargs
    )

    # 输出文件
    with open(output_path, "w") as f:
        for id,response in result_generator:
            json.dump({
                "id": id,
                "response": response
            }, f, ensure_ascii=False, indent=4)
            f.write("\n")
    

def test():
    # 测试数据
    data_list = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "1，3，11，13，21...., 下一个数字是多少"},
        ]
    client = Client()
    
    res = client.complete(data_list,stream=False)
    # 这里的print有什么问题吗，为什么print不出来

    print(res)
    for _ in res:
        print(_)
        
    

if __name__ == "__main__":
    
    data_process(
        output_path="../output/vllm_openai_client_multithread.json",
    )
    # test()