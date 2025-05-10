# mcp tools
## 1. some related docs
    https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm
    https://docs.vllm.ai/en/v0.7.1/serving/openai_compatible_server.html
    https://openai.github.io/openai-agents-python/mcp/
    https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/clients/simple-chatbot/mcp_simple_chatbot/main.py

## 一些疑问
    1. 如果tools太多，不适合mcp（上下文会超）
        如果tools太少，没必要用mcp（自己写一个tools文件）


## 5.8 工作计划
    1. 探索怎么把已有的mcp server 整合到py文件中
        方法一：Python 写StdioServerParameters
        方法二： 使用json文件