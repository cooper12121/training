# This is for function call 

""" 
1. some related docs
    https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm
    https://docs.vllm.ai/en/v0.7.1/serving/openai_compatible_server.html

2. notices
    1. 注意封装tool的返回值，直接能够作为content,避免进一步编辑 to en: Ensure tools return values are directly usable as content to avoid further editing.

3. 为了保持统一，所有的server都使用mcp_server.py中的函数
"""
from transformers.utils import get_json_schema


def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    # A real function should probably actually get the temperature!
    return {
        "location": location,
        "unit": unit,
        "temperature": 22.0
        }  

def get_current_wind_speed(location: str) -> float:
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current wind speed at the given location in km/h, as a float.
    """
    return {
        "location": location,
        "wind_speed": 15.0
        }

tools = [get_current_temperature, get_current_wind_speed]

# schema = get_json_schema(get_current_temperature)
# print(schema)