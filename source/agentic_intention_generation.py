import json
from tools.gpt_inference import do_inference
import random
import re
def prompt_construction(food, persona):
    prompt = f"""
    You are generating a concise, realistic user situation linking a food item to a user persona.

    ### Persona
    {persona}

    ### Food
    {food}

    ### Task
    Create a short real-life context (2–5 sentences) showing why and when this user would consider eating this food.
    Reflect their goals, habits, and potential concerns based on the ingredients and nutrition profile.

    Output in this JSON format:
    {{
      "UserSituation": {{
        "persona_id": "",
        "ingredients": <from food_json.ingredients>,
        "context": "1–2 sentences about what the user is doing or feeling.",
        "motivation": "1 sentence on why they consider this food.",
        "concern": "1 short note on what they might worry about (e.g., additives, calories, sodium)."
      }}
    }}
    """

    return prompt


import json
from typing import List, Union, Optional

OPEN_TO_CLOSE = {"{": "}", "[": "]"}
CLOSE_TO_OPEN = {"}": "{", "]": "["}
OPENERS = set(OPEN_TO_CLOSE.keys())
CLOSERS = set(CLOSE_TO_OPEN.keys())

def _balanced_json_slices(text: str) -> List[str]:
    """
    Scan text and return substrings that look like balanced JSON objects/arrays.
    Handles quotes and escapes so braces inside strings won't confuse it.
    """
    slices = []
    stack = []
    start_idx: Optional[int] = None
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            # Inside string: ignore everything else
            continue

        # Not in string
        if ch == '"':
            in_str = True
            continue

        if ch in OPENERS:
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch in CLOSERS and stack:
            if CLOSE_TO_OPEN[ch] == stack[-1]:
                stack.pop()
                if not stack and start_idx is not None:
                    slices.append(text[start_idx:i+1])
                    start_idx = None
            else:
                # Mismatched bracket; reset to be safe
                stack.clear()
                start_idx = None

    return slices

def extract_json_from_output(output_text: str) -> Union[dict, list, None]:
    """
    Extract and parse JSON from a model's raw output.
    Returns:
      - dict or list if one JSON block found
      - list of parsed objects if multiple JSON blocks found
      - None if nothing valid found
    """
    candidates = _balanced_json_slices(output_text)
    parsed = []
    for blob in candidates:
        try:
            parsed.append(json.loads(blob))
        except json.JSONDecodeError:
            # Try a light cleanup for code fences or trailing commas
            cleaned = blob.strip()
            if cleaned.startswith("```"):
                # strip triple backticks wrappers if present
                cleaned = cleaned.strip("`")
                # Often it's like: ```json\n{...}\n```
                # Try to find the first { or [ again inside
                inner = cleaned.find("{")
                inner_alt = cleaned.find("[")
                inner_idx = min(x for x in [inner, inner_alt] if x != -1) if (inner != -1 or inner_alt != -1) else -1
                if inner_idx != -1:
                    cleaned = cleaned[inner_idx:]
            try:
                parsed.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue

    if not parsed:
        return None
    return parsed[0] if len(parsed) == 1 else parsed

total_results = []

with open('PATH_TO_USER_INPUT_OUTPUT_DATA') as jsonfile:
    total_foods = json.load(jsonfile)

with open('PATH_TO_LIST_OF_PERSONAS') as jsonfile:
    total_personas = json.load(jsonfile)

for food_num in range(len(total_foods)):
    print(f"Current food num: {food_num}")
    random.seed(food_num)
    food_details = random.sample(total_foods, 1)[0]
    persona_details = random.sample(total_personas, 1)[0]
    prompt = prompt_construction(food_details, persona_details)
    result = do_inference(prompt)
    json_dicts = extract_json_from_output(result)
    if json_dicts:
        json_dicts["UserSituation"]['food_id'] = food_details['food_id']
        total_results.append(json_dicts)

    with open('PATH_TO_OUTPUT_DIR', 'w') as jsonfile:
        json.dump(total_results, jsonfile, indent=4)
