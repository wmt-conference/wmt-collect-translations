import os
from tools.errors import FINISH_STOP, FINISH_LENGTH


CLIENT = None
def lazy_get_client():
    global CLIENT
        
    if CLIENT is None:
        from openai import OpenAI

        assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"
        CLIENT = OpenAI()
    return CLIENT


def process_with_openai_gpt4_1(request, max_tokens=None, temperature=None):  
    if max_tokens is None:
        max_tokens = 32768
    if temperature is None:
        temperature = 0.0
    return openai_call(request, "gpt-5.1", temperature=temperature, max_tokens=max_tokens)


def openai_call(request, model, temperature=0.0, max_tokens=None):
    client = lazy_get_client()
    import openai

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": request['prompt']}
            ],
            max_completion_tokens=max_tokens,
            reasoning_effort="none",
        )    
    except (openai.BadRequestError, openai.APITimeoutError) as e:
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        raise e

    if response.choices[0].finish_reason == "length":
        finish_reason = FINISH_LENGTH
    elif response.choices[0].finish_reason == "stop":
        finish_reason = FINISH_STOP
    else:
        return None

    return response.choices[0].message.content, {
        "raw_response": response.model_dump(mode="json"),
        "model": response.model,
        "temperature": None,
        "reasoning_trace": None,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "thinking_tokens": 0,
        "finish_reason": finish_reason
    }

