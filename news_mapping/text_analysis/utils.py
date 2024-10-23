import pandas as pd

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
        return None

def extract_inside_braces(s):
    start = s.find("{")
    end = s.find("}")
    if start >= 0 and end > start:
        return s[start : end + 1].strip()
    else:
        return None

def additional_filter(dataframe: pd.DataFrame, match_words: list) -> pd.DataFrame:
    """ """
    dataframe = dataframe[
        dataframe["source"].apply(contains_any_word, words=match_words)
    ].reset_index(drop=True)
    dataframe["source"] = dataframe["source"].apply(
        lambda x: if_contains_assign(x, match_words)
    )
    return dataframe


