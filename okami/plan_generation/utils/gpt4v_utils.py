import re
import numpy as np
import json
import cv2
import base64

def json_parser(txt):
    '''
    parse json from txt
    '''
    # Regular expression to extract content within ```json and ```
    pattern = r'```json\s*(\{.*?\})\s*```'  # Non-greedy matching for JSON object

    json_data = None
    match = re.findall(pattern, txt, re.DOTALL)  # DOTALL to match across lines
    if match:
        json_str = match[-1]
        try:
            # Parse the JSON string into a Python dictionary
            json_data = robust_json_loader(json_str)
            print("Extracted JSON data:", json_data)
        except:
            print("Error decoding JSON from vlm response.")
            raise ValueError("Error decoding JSON from vlm response.")
    else:
        print("No JSON found in the text.")
    
    return json_data

def robust_json_loader(json_str):
    '''
    Load JSON string robustly
    '''
    try:
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        error_message = str(e)
        if error_message.startswith("Invalid \\escape"):
            json_str = fix_invalid_escape(json_str, error_message)
        if error_message.startswith(
            "Expecting property name enclosed in double quotes"
        ):
            json_str = add_quotes_to_property_names(json_str)
        
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            try:
                brace_index = json_str.index("{")
                json_str = json_str[brace_index:]
                last_brace_index = json_str.rindex("}")
                json_str = json_str[: last_brace_index + 1]
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise e

def fix_invalid_escape(json_str: str, error_message: str) -> str:
    while error_message.startswith("Invalid \\escape"):
        bad_escape_location = extract_char_position(error_message)
        json_str = json_str[:bad_escape_location] + json_str[bad_escape_location + 1 :]
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            error_message = str(e)
    return json_str

def extract_char_position(error_message: str) -> int:
    """Extract the character position from the JSONDecodeError message.
    Args:
        error_message (str): The error message from the JSONDecodeError
          exception.
    Returns:
        int: The character position.
    """
    import re

    char_pattern = re.compile(r"\(char (\d+)\)")
    if match := char_pattern.search(error_message):
        return int(match[1])
    else:
        raise ValueError("Character position not found in the error message.")

def add_quotes_to_property_names(json_string: str) -> str:
    """
    Add quotes to property names in a JSON string.
    Args:
        json_string (str): The JSON string.
    Returns:
        str: The JSON string with quotes added to property names.
    """

    def replace_func(match):
        return f'"{match.group(1)}":'

    property_name_pattern = re.compile(r"(\w+):")
    corrected_json_string = property_name_pattern.sub(replace_func, json_string)

    try:
        json.loads(corrected_json_string)
        return corrected_json_string
    except json.JSONDecodeError as e:
        raise e

def encode_imgs_from_path(vlm, img_paths):
    '''
        Read in image from img_paths, and convert to base64 string.
    '''
    base64_img_list = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = vlm.preprocess_image(img, is_rgb=False, encoding='jpg')
        base64_img_list.append(img)
    return base64_img_list