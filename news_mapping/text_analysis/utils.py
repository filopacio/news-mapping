import pandas as pd
import tiktoken

def contains_any_word(row, words):
    return any(word in row.split() for word in words)


def if_contains_assign(row, words, keywords):
    word_dict = {}
    for i in range(len(words)):
        word_dict[keywords[i]] = words[i]
    for keyword in keywords:
        if keyword in row:
            return word_dict[keyword]
    return None

def evaluate_string(expression):
    try:
        result = eval(expression)
        return result
    except Exception as e:
        print(f"Error evaluating expression '{expression}': {e}")
        return "{}"

def extract_inside_braces(string: str) -> str:
    bracket_count = 0
    start_index = -1
    end_index = -1

    # Iterate through the string to find the outermost brackets
    for i, char in enumerate(string):
        if char == '{':
            if bracket_count == 0:  # First opening bracket
                start_index = i
            bracket_count += 1
        elif char == '}':
            bracket_count -= 1
            if bracket_count == 0:  # Last closing bracket
                end_index = i + 1  # Include the closing bracket
                break

    # Extract the substring that contains the list
    if start_index != -1 and end_index != -1:
        return string[start_index:end_index]
    else:
        return "{}"

def additional_filter(dataframe: pd.DataFrame, match_words: list) -> pd.DataFrame:
    """ """
    dataframe = dataframe[
        dataframe["source"].apply(contains_any_word, words=match_words)
    ].reset_index(drop=True)
    dataframe["source"] = dataframe["source"].apply(
        lambda x: if_contains_assign(x, match_words)
    )
    return dataframe

def filter_newspapers(dataframe: pd.DataFrame, accepted_newspapers: list) -> pd.DataFrame:
    """
    """
    accepted_words_sets = [set(name.lower().split()) for name in accepted_newspapers]

    def has_common_words(news_name):
        news_words = set(news_name.lower().split())
        return any(news_words & accepted_set for accepted_set in accepted_words_sets)

    return dataframe[dataframe["newspaper"].apply(has_common_words)]


def map_incomplete_to_full_names(names) -> list:
    """
    """
    surname_to_fullname = {}

    # First pass: collect full names
    for name in names:
        parts = name.strip().split()
        if len(parts) >= 2:
            # Full name case (first and last name)
            surname = parts[-1]  # Last part is the surname
            surname_to_fullname[surname] = name  # Map surname to full name

    # Second pass: replace incomplete names with full names when possible
    mapped_names = []
    for name in names:
        parts = name.strip().split()
        if len(parts) == 1:
            # Only surname provided
            surname = parts[0]
            # Replace with full name if found
            full_name = surname_to_fullname.get(surname, name)
            mapped_names.append(full_name)
        else:
            # Full name already provided
            mapped_names.append(name)

    return mapped_names


def calculate_token(string: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(string))


def clean_json_string(input_string):
    replacements = {
        '‘': "'",
        '’': "'",
        '“': '"',
        '”': '"',
        '\u00A0': ' ',
        '\u200B': ''
    }

    for old, new in replacements.items():
        input_string = input_string.replace(old, new)

    input_string = ''.join(c for c in input_string if ord(c) >= 32)

    return input_string
