import hashlib
import json
from typing import Any
import regex


def extract_valid_json_substring(string: str) -> str:
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
    return ""


def calculate_md5(input_data: Any) -> str:
    if isinstance(input_data, dict):
        # Convert dictionary to a JSON-formatted string
        input_str = json.dumps(input_data, sort_keys=True).encode("utf-8")
    elif hasattr(input_data, "read"):
        # Read binary data from file-like object
        md5_hash = hashlib.md5()
        for chunk in iter(lambda: input_data.read(4096), b""):
            md5_hash.update(chunk)
        return md5_hash.hexdigest()
    elif isinstance(input_data, list):
        md5_hash = hashlib.md5()
        for item in input_data:
            md5_hash.update(calculate_md5(item).encode("utf-8"))
        return md5_hash.hexdigest()
    else:
        # Convert other input to byte string
        input_str = str(input_data).encode("utf-8")

    # Calculate MD5 hash of byte string
    md5_hash = hashlib.md5(input_str)
    return md5_hash.hexdigest()
