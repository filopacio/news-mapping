import time
from groq import Groq


def obtain_topics_and_person(
    text: str,
    api_key: str,
    topics_to_scrape: None,
    max_tokens: int = 1024,
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
        1.5
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
Sei un analista di notizie. Dal testo fornito, che è un articolo di notizie ottenuto tramite scraping di HTML,
devi individuare:
1. Il testo dell'articolo, escludendo tutte le altre parole appartenenti ad un sito web ma non all'articolo.
2. L'UNICO argomento principale discusso nell'articolo.
3. Tutti i nomi propri di personaggi pubblici menzionati.
Non aggiungere nessun altro commento o testo oltre all'oggetto JSON. Segui **rigorosamente** queste istruzioni.

Il risultato dovrà essere **esclusivamente** un oggetto JSON con la seguente struttura:

{{
  "text": "<string> (SOLO il testo dell'articolo, separato dal resto delle parole non appartenenti all'articolo)",
  "topic": "<string> (unico argomento principale dall'articolo, lasciarlo vuoto se nessun argomento è valido)",
  "persons": ["<string> (lista dei nomi propri di personaggi pubblici menzionati, se presenti, altrimenti lista vuota)"]
}}

{topics_string}

Ecco il testo: {text}.
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
        1.5
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
