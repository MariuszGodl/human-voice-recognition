import os
import openai
import json
import tiktoken




def count_tokens(messages, total_response_tokens,  model="gpt-4",):
    """
    Count the tokens used by a list of messages for a given model.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        # Each message format adds 4 tokens, plus 1 per key/value string
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # Every reply adds 2 tokens

    # File where the total tokens are stored
    file_path = "used_tokens.txt"

    # Read current total tokens
    try:
        with open(file_path, "r") as f:
            total_tokens = int(f.read().strip())
    except FileNotFoundError:
        total_tokens = 0  # If file doesn't exist, start from 0

    # Add new tokens
    total_tokens += num_tokens + total_response_tokens

    # Save updated total tokens back to file
    with open(file_path, "w") as f:
        f.write(str(total_tokens))



def check_if_it_is_real_word(word, language='Polish'):
    """
    Checks if a word exists in the specified language using ChatGPT.
    Returns True if it exists, False otherwise.
    """
    json_file = "words.json"

    # Load existing JSON data
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    # Check cache first
    if word in data:
        return data[word]["is_real"], data[word]["corrected_word"]

    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = (
        f"Does the word '{word}' exist in {language}? Ignore Polish characters (e.g., 'moze' instead of 'może')."
        " Words with missing letters (e.g., 'kacka') should be 'no'. Answer only 'yes' or 'no'."
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=3,
        temperature=0
    )

    count_tokens(messages, response.usage.total_tokens, model="gpt-4")



    # Access content as an attribute, not like a dict
    answer = response.choices[0].message.content.strip().lower()

    #append the word into words.jeson with corresponding bool

    is_real = answer == "yes"

    # Load existing JSON data

    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    # Update JSON with new word
    data[word] = {
        "word": word,
        "is_real": is_real,
        "corrected_word": None
    }

    # Save back to file
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return is_real, None
