import os
import logging
from tools.errors import FINISH_LENGTH, FINISH_STOP

CLIENT = None
def lazy_get_client():
    global CLIENT
    
    if CLIENT is None:
        from mistralai import Mistral

        assert "MISTRAL_API_KEY" in os.environ, "Please set the MISTRAL_API_KEY environment variable"

        CLIENT = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    return CLIENT

def process_with_mistral_medium(request, max_tokens=None, temperature=None):
    if max_tokens is None:
        max_tokens = 8192
    if temperature is None:
        temperature = 0.0
    return process_with_mistral(request, "mistral-medium-latest", max_tokens=max_tokens, temperature=temperature)

# setting max_tokens to None uses maximum allowed tokens of given model
def process_with_mistral(request, model, max_tokens=None, temperature=0.0):
    client = lazy_get_client()

    messages = [{"role": "user", "content": request['prompt']}]

    try:
        response = client.chat.complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

    if response.choices[0].finish_reason == "stop":
        finish_reason = FINISH_STOP
    elif response.choices[0].finish_reason == "length":
        finish_reason = FINISH_LENGTH
    else:
       return None

    return response.choices[0].message.content, {
        "raw_response": response.model_dump(mode="json"),
        "model": response.model,
        "temperature": temperature,
        "reasoning_trace": None,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "thinking_tokens": 0,
        "finish_reason": finish_reason
    }

