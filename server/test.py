# from transformers import AutoModelForCausalLM, AutoTokenizer
# from dotenv import load_dotenv
# import os
# load_dotenv("/Users/qianggao/project/intern/training/server/.env")

# model_name = os.getenv("MODEL_PATH")

# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# # prepare the model input
# prompt = "Give me a short introduction to large language model. /think"
# prompt = "Give me a short introduction to large language model. /think"
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     # enable_thinking=True # Switch between thinking and non-thinking modes. Default is True.
# )
# print(text)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# # conduct text completion
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
# all_output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
# all_content = tokenizer.decode(all_output_ids, skip_special_tokens=True).strip("\n")
# print("all content:", all_content)
# # parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
# print("content:", content)


# import json
# from transformers import AutoTokenizer,AutoModelForCausalLM
# from test import tools



# messages = [
#   {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
#   {"role": "user", "content": "Hey, what's the temperature and wind in Paris right now?"}
# ]
# documents = [
#     {
#         "title": "The Moon: Our Age-Old Foe", 
#         "text": "Man has always dreamed of destroying the moon. In this essay, I shall..."
#     },
#     {
#         "title": "The Sun: Our Age-Old Friend",
#         "text": "Although often underappreciated, the sun provides several notable benefits..."
#     }
# ]

# # tool_call = {"name": "get_current_temperature", "arguments": json.dumps({"location": "Paris, France", "unit": "celsius"})} # 注意这里要修改为字符串，或者修改模版文件'```json' + '\\n' + (tool['function']['arguments'] | tojson) + '\\n' + '```'
# # messages.append({"role": "assistant", "content":None ,"tool_calls": [{"type": "function", "function": tool_call}]})
# # messages.append({"role": "tool", "name": "get_current_temperature", "content": "22.0"})

# ckpt = "/Users/qianggao/project/intern/ckpt/Hermes-2-Pro-Llama-3-8B"
# tokenizer = AutoTokenizer.from_pretrained(ckpt)
# # model  = AutoModelForCausalLM.from_pretrained(ckpt, device_map="cuda:0", torch_dtype="auto")


# inputs = tokenizer.apply_chat_template(
#     messages,
#     tools=tools,
#     documents=documents,
#     # chat_template="rag",
#     tokenize=False,
#     # return_tensors="pt",
#     # return_dict=True,
#     # add_generation_prompt=True,
# )
# print(inputs)
# inputs = {k: v.to(model.device) for k, v in inputs.items()}
# out = model.generate(**inputs, max_new_tokens=512,pad_token_id =tokenizer.eos_token_id)
# print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))


def test(): 
    import json5
    with open("/Users/qianggao/project/intern/training/server/mcp_server_config.json5", "r") as f:
        config = json5.load(f)
    print(config)
    with open("/Users/qianggao/project/intern/training/server/mcp_server_config.json", "w") as f:
        json5.dump(config, f, indent=4)

test()