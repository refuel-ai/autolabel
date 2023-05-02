import json
import regex


def extract_valid_json_substring(string):
    pattern = (
        r"{(?:[^{}]|(?R))*}"  # Regular expression pattern to match a valid JSON object
    )
    match = regex.search(pattern, string)
    if match:
        json_string = match.group(0)
        try:
            json.loads(json_string)
            return json_string
        except ValueError:
            pass
    return None
