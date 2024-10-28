import time
from groq import Groq


from news_mapping.text_analysis.utils import extract_inside_braces, evaluate_string

import pandas as pd
from tqdm import tqdm


def get_newspaper_topics_persons(
        dataframe: pd.DataFrame,
        api_key: str,
        query: str,
        topics_to_scrape: None or list,
        model: str,
        batch_size: int = 10
) -> pd.DataFrame:
    """
    Concatenates text from the DataFrame every `batch_size` rows and applies
    the obtain_topics_and_person function to the concatenated text.
    """

    # Initialize the new column
    results = []

    for i in tqdm(range(0, len(dataframe), batch_size)):
        concatenated_text = ""
        for j in range(i, min(i + batch_size, i + len(dataframe.iloc[i: i + batch_size, :]))):
            concatenated_text += f"""
            ------------------
            newspaper: {dataframe.loc[j, "newspaper"]}
            text: {dataframe.loc[j, "text"]}
            ------------------
            """

        # Call the external function with the concatenated text
        result = retrieve_from_articles(
            text=concatenated_text,
            api_key=api_key,
            query=query,
            topics_to_scrape=topics_to_scrape,
            model=model
        )

        # Process the result
        result = extract_inside_braces(result)
        result = evaluate_string(result)

        results += result

    dataframe["newspaper_topics_persons"] = results

    return dataframe


def retrieve_from_articles(
    text: str,
    api_key: str,
    query: str,
    topics_to_scrape: None or list,
    max_tokens: int = 8000,
    model: str = "llama3-70b-8192",
) -> str:
    """
    Retrieve topics discussed and people mentioned in the text provided
    :param text: text of the article
    :param api_key: api key. only grow supported so far
    :param topics_to_scrape: to facilitate the work to LLM, a set of topics is provided a priori.
    :param max_tokens: max tokens
    :param model: model adopted
    :return: the LLM call output.
    """
    time.sleep(
        1
    )  # to avoid reaching maximum requests per seconds and tokens per minute
    client = Groq(api_key=api_key)

    if topics_to_scrape:
        topics_string = """L'argomento deve appartenere **solamente** ad una delle seguenti categorie: {topics_to_scrape}. 
                            Se nessuno degli argomenti è correttamente riflesso nell'articolo, lascia il campo 
                           'topic' vuoto. """
    else:
        topics_string = ""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Sei un analista di notizie."},
            {
                "role": "user",
                "content": f"""
Data la seguente lista di pagine web di giornali, per ogni articolo, identifica quanto segue:
1. Il nome del giornale che ha scritto l'articolo
2. Il testo dell'articolo nella pagina riguardante {query}, rimuovendo tutte le alte parole della pagina web
 non appartententi all'articolo (come annunci, pubblicità, altri articoli irrilevanti).
3. Estrai l'argomento principale discusso nel testo (deve essere uno e uno solo).
4. Identifica tutti i nomi propri di persone menzionate nel testo.
Per ogni articolo, restituisci i risultati sotto forma di lista di JSON come descritto qui sotto:
[<{{
  "newspaper": "<nome_del_giornale>",
  "text": ""<il testo dell'artiolo riguardante {query}>
  "topics": "<argomento_principale in italiano>",
  "persons": ["<nome_persona1>", "<nome_persona2>", ...]
}}>, <json 2>, ...]

{topics_string}
Restituisci solo e soltanto la lista richiesta senza aggiungere nessun altro commento

Ecco la lista:
{text}
""",
            },
        ],
        model=model,
        max_tokens=max_tokens,
    )

    return chat_completion.choices[0].message.content


def summarize_text(
    text: str, api_key: str, max_tokens: int = 200, model: str = "llama3-70b-8192"
) -> str:
    """
    Summarizes a given piece of text using a language model.

    Args:
        text (str): The text to be summarized.
        api_key (str): The API key for accessing the language model.
        model (str): The model to be used for summarization. Default is "gpt-4".
        max_tokens (int): The maximum number of tokens in the summary. Default is 100.
    Returns:
        str: The summarized text.
    """
    time.sleep(
        0.8
    )  # to avoid reaching maximum requests per seconds and tokens per minute
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Sei un analista di notizie."},
            {
                "role": "user",
                "content": f"""Riassumi il seguente testo in 500 parole.
                               Restituisci SOLO il testo riassunto senza
                               commenti aggiuntivi. E' molto importante che segui
                               fedelmente queste istruzioni. Ecco il testo: {text}""",
            },
        ],
        model=model,
        max_tokens=max_tokens,
    )

    return chat_completion.choices[0].message.content
