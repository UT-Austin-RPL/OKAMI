'''
    Helpful tutorials: https://cookbook.openai.com/
'''
import os
import sys
import base64
import cv2
import numpy as np
from typing import List, Union
from termcolor import colored
from easydict import EasyDict
from openai import OpenAI

class GPT4V:
    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
    ):
        super().__init__()
        # only tested with gpt-4-turbo but most of the boilerplate code should work with other models as well.
        assert model_name in ["gpt-4-turbo"], f"Model {model_name} not supported."
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
        self.begin_new_dialog()
    
    def begin_new_dialog(self):
        self.history_msgs = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]

    @property
    def available_encodings(self):
        # additional available encodings are: 'gif', 'webp'
        return ['jpg', 'png', 'jpeg']

    def verify_msg(self, msgs: List[dict]):
        for msg in msgs:
            assert msg['role'] in ['user', 'system', 'assistant'], f"role {msg['role']} not recognized."
            for content in msg['content']:
                # GPT4V only supports text for system and assistant at the moment
                if (msg['role'] == 'system') or (msg['role'] == 'assistant'):
                    assert isinstance(content, str), f"content {content} not recognized for role {msg['role']}."
                elif msg['role'] == 'user':
                    assert content['type'] in ['text', 'image_url'], f"type {content['type']} not recognized."
                    if content['type'] == 'image_url':
                        assert isinstance(content['image_url'], dict), f"image_url {type(content['image_url'])} not recognized."
                    elif content['type'] == 'text':
                        assert isinstance(content['text'], str), f"text {content['text']} not recognized."
                    else:
                        raise ValueError(f"type {content['type']} not recognized.")
        return True

    def check_msgs_mem(self, msgs: list):
        # calculate the memory of the msgs
        total_mem = sys.getsizeof(msgs) / 1024 / 1024
        assert total_mem < 20, "total msg should be less than 20Mb"
        return total_mem

    def preprocess_image(self, img: np.ndarray, is_rgb: bool, encoding: str = 'jpg'):
        '''
            Convert the np.ndarray image to base64 string.
        '''
        assert encoding in self.available_encodings, f"Encoding {encoding} not supported."
        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.'+encoding, img)
        return base64.b64encode(buffer).decode('utf-8')

    def generate_msg(
            self,
            text_prompt_list: List[str],
            base64_img_list: Union[List[str], List[List[str]]],
            role_list: List[str] = None,
            encoding_list: List[str] = None,
            detail_list: List[str] = None,
        ):
        '''
            Given a list of text prompts and base64 images, wrap them in the required format.
            - We assume that each text is corresponding to single image. This can be tweaked by adding more content with type 'image_url' in the msg.
            - You can also specify the role of the user, system, or assistant. For system and assistant, only text is supported.
        '''
        assert len(text_prompt_list) == len(base64_img_list), \
            f"Length of text_prompt_list {len(text_prompt_list)}, base64_img_list {len(base64_img_list)} not equal."
        if role_list is None:
            role_list = ["user"] * len(text_prompt_list)
        if encoding_list is None:
            encoding_list = ["jpg"] * len(base64_img_list)
        if detail_list is None: # detail_list can be low or high.
            detail_list = ["high"] * len(text_prompt_list)
        msgs = self.history_msgs.copy()
        for i in range(len(text_prompt_list)):
            msg = {
                "role": role_list[i],
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt_list[i]
                    }
                ]
            }
            if (role_list[i] == "user") and (base64_img_list[i] is not None):
                if isinstance(base64_img_list[i], list):
                    for img in base64_img_list[i]:
                        msg['content'].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{encoding_list[i]};base64,{img}",
                                "detail": detail_list[i]
                            }
                        })
                else:
                    msg['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{encoding_list[i]};base64,{base64_img_list[i]}",
                            "detail": detail_list[i]
                        }
                    })

            msgs.append(msg)

        self.check_msgs_mem(msgs)
        return msgs

    def generate_response(
            self,
            msgs: List[dict],
            max_tokens: int,
            temperature: float = 0,
            fake_response: bool = False,
        ):
        assert self.verify_msg(msgs), "msgs not verified"
        if fake_response:
            # create a fake response
            fake_response = EasyDict({
                "choices": [
                    {
                        "finish_reason": "fake",
                        "message": {
                            "content": "fake"
                        }
                    }
                ]
            })
            return fake_response
        try:
            print("generating response...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                # response_format={"type": "json_object"}, # Not supported in GPT-4 Vision
            )
        except Exception as e:
            print(colored(f"[ERROR] {e}", 'red'))
            import pdb; pdb.set_trace()
            return None

        if response.choices[0].finish_reason == 'length':
            print(colored(f"[WARNING] Output may be incomplete due to token limit. Consider increasing your max_tokens", 'red'))
        elif response.choices[0].finish_reason == 'fake':
            print(colored(f"[WARNING] Fake response detected.", 'red'))
        elif response.choices[0].finish_reason == 'content_filter':
            print(colored(f"[WARNING] content was omitted due to a flag from openai content filters", 'red'))
        elif response.choices[0].finish_reason == 'stop':
            pass
        
        self.history_msgs = msgs.copy() # add to history
        self.history_msgs.append({ # add the response to history
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        return self.cleanup_response(response.choices[0].message.content), response

    def cleanup_response(self, text_response: str):
        '''
            Given the response string, cleanup the response and return the cleaned response.
            Custom cleanup can be done here, e.g., json parsing.
        '''
        # some clean up
        return text_response
    
    def run(self, 
            text_prompt_list: List[str],
            base64_img_list: Union[List[str], List[List[str]]],
            max_tokens: int = 500,
            role_list: List[str] = None,
            encoding_list: List[str] = None,
            detail_list: List[str] = None,
        ):
        """
        high-level API: given a list of text prompts and base64 images, generate a response.
        """
        msgs = self.generate_msg(text_prompt_list, base64_img_list, role_list, encoding_list, detail_list)
        text_response, full_response = self.generate_response(msgs, max_tokens=max_tokens)
        return text_response

if __name__ == '__main__':
    img = 'test.jpg'
    img = cv2.imread(img)
    vlm = GPT4V()
    img = vlm.preprocess_image(img, is_rgb=False, encoding='png')
    text_prompt_list = ["Can you describe the image in few words?"]
    base64_img_list = [img]
    role_list = ["user"]
    msgs = vlm.generate_msg(text_prompt_list, base64_img_list, role_list)
    vlm.check_msgs_mem(msgs)
    text_response, full_response = vlm.generate_response(msgs, max_tokens=100)
    # print(full_response)
    print(text_response)
