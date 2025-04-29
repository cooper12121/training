# from openai import OpenAI
# # Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"
# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )
# messages = None
# chat_response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-1.5B-Instruct",
#     messages=messages,
#     extra_body = {
#         "top_k":50,
#         "repetition_penalty":1.05,
#         # "stop_token_ids":
#     }
    
# )
# print("Chat response:", chat_response)   

import openai
import requests
import json
import logging
import time
import random
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import os

# 获取变量
model_path = os.getenv("MODEL_PATH")
openai_api_key = os.getenv("API_KEY")
openai_api_base = os.getenv("API_BASE")

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
REQUEST_UNSAFE_STR = 'error_code=content_filter'
class Client(object):
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 api_key=openai_api_key,
                 url: str = openai_api_base,
                 ):
        self.api_key = api_key
        self.model_name = model_name
        self.url = url
        # if logdir is None:
        #     logdir = ROOT / 'data/llm_logs'
        # self.logdir = Path(logdir)

    def __call__(self, *args, **kwargs):
        return self.complete(*args, **kwargs)

    def complete(self, messages: list[dict],
                 content_only=False,
                 stream=False,
                 **kwargs) -> Union[list[str], dict, requests.Response]:
        """包括拿到response之后的处理

        :param messages: list[Dict]
        :param content_only: bool, 是否只提取文本；
            if True，返回 list[str]list
        :param stream: bool, 是否流式输出；
            if True，返回生成器（Response），需要额外解码
        :param kwargs:
        :return:
        """

        resp = self._complete(messages, stream=stream, **kwargs)
        if stream:
            return resp  # setting ``stream=True`` gets the response (Generator)

        if content_only:
            if 'choices' in resp:
                choices = resp['choices']
                return [x['message'].get('content', None) for x in choices]
        return resp

    def _complete(self,
                  messages: list[dict],
                  n=1,
                  max_tokens=1500,
                  temperature: float = 0.8,
                  **kwargs) -> requests.Response:
        """只负责请求api，返回response，不做后处理

        :param messages:
        :param content_only:
        :param n: the number of candidate model generates
        :param max_tokens: max number tokens of the completion (prompt tokens are not included)
        :param kwargs:
        :return:
        """
        openai.api_key = self.api_key
        openai.api_base = self.url
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            n=n, temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response

def process_msg(model_name: str,
               messages: list[dict],
               max_try: int = 3,
               response_only: bool = False,
               id: str = None,
               **generate_kwargs) -> tuple[str, list[dict],requests.Response ,str]:
        """ 
        请求单条响应
        :param model_name: str, 模型名称
        :param messages: list[Dict], 消息列表
        :param max_try: int, 最大尝试次数
        :param response_only: bool, 是否返回结构化响应，false只返回相应文本
        :param id: str, 消息ID
        :return: tuple, (id, messages, response, res)"""
        if messages[-1].get("role")=="user" and messages[-1].get("content")==None:
            return (id, messages, None, None)
        
        for i in range(max_try):
            try:
                llm = Client(model_name=model_name)
                if response_only:
                    print("***respnse only***")
                    resp = llm._complete(messages,**generate_kwargs)
                    return (id, messages, resp, None)
                else:
                    res = llm.complete(messages, content_only=True,**generate_kwargs)
                    return (id, messages, None, res[0])
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
        model_name: str,
        max_workers: int = 1,
        max_try: int = 3,
        response_only: bool = False,
        **generate_kwargs,
        
    ):
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor: # 5 is just an example, adjust according to your needs
        futures = {executor.submit(process_msg, id=id, msg=msg,max_try=max_try,model_name=model_name,response_only = response_only ,**generate_kwargs): id for id, msg in tqdm(enumerate(data_list))}

        count = 0
        for future in tqdm(as_completed(futures), total=len(futures), ncols=70):
            try:
                id, messages, resp, res_text = future.result()
                logging.info(f"prompt:\n{messages}")
                
                if response_only:
                    # 返回完整的结构化响应
                    logging.info(f"response:\n{resp.json()}")
                    data_list[id].append(resp.json())
                    yield data_list[id]

                else:
                    # 返回文本响应内容
                    logging.info(f"reponse:\n{res_text}\n")
                    data_list[id].append(
                        {"role": "assistant", "content": res_text}
                    )
                    yield data_list[id]
                
            except Exception as e:
                print("error:",e)
                count+=1
            
    print(f"total error number:{count}")


def load_dataset():
    import os,sys
    from datasets import load_dataset
    path = "google-research-datasets/natural_questions"
    subset = "dev"
    split = "validation"
    dataset_name = path.split('/')[-1]
    dataset = load_dataset(path,subset,split=split,cache_dir=None)
    dataset.to_json(f"/Users/qianggao/project/intern/rag/dataset/{dataset_name}/{subset}/{split}.json")


def Data_Process(intput_file:str, output_file:str):
    



    # ToDO: 这里需要根据实际数据进行处理,datalist
    data_list = None

    generate_kwargs = {
        "n":1,
        "max_tokens":1500,
        "repetition_penalty": 1.05,
        "temperature": 0.8
    }
    max_workers = 1

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    result_generator = LLM_Caller(
        data_list,
        model_name=model_name,
        max_workers=max_workers,
        response_only=True,
        **generate_kwargs
    )

    # 输出文件
    with open(output_file, "w") as f:
        for result in result_generator:
            json.dump(result, f)
            f.write("\n")