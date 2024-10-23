import time
from groq import Groq


def obtain_topics(
    text: str,
    api_key: str,
    topics_to_scrape: list = None,  # Optional list of topics
    max_tokens: int = 1024,
    model: str = "llama3-70b-8192",
) -> str:
    """
    Retrieve the main topic discussed in the text provided.
    :param text: text of the article
    :param api_key: API key. Only Groq supported so far.
    :param topics_to_scrape: Optional list of topics to restrict the model to.
    :param max_tokens: max tokens for the LLM call
    :param model: model adopted
    :return: the LLM call output with the main topic.
    """
    client = Groq(api_key=api_key)

    if topics_to_scrape:
        # If topics are provided, restrict the analysis to those topics
        topics_string = f"L'argomento deve appartenere **solamente** ad una delle seguenti categorie: {topics_to_scrape}."
    else:
        # No restriction on topics if none are provided
        topics_string = ""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Sei un analista di notizie."},
            {
                "role": "user",
                "content": f"""
Sei un analista di notizie. Dal testo fornito, che è un articolo di notizie ottenuto tramite scraping di HTML,
devi individuare:
1. L'UNICO argomento principale discusso nell'articolo.

Non aggiungere nessun altro commento o testo oltre all'oggetto JSON. Segui **rigorosamente** queste istruzioni.

Il risultato dovrà essere **esclusivamente** un oggetto JSON con la seguente struttura:

{{
  "topic": "<string> (unico argomento principale dall'articolo, lasciarlo vuoto se nessun argomento è valido)"
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


def obtain_persons(
    text: str,
    api_key: str,
    max_tokens: int = 1024,
    model: str = "llama3-70b-8192",
) -> str:
    """
    Retrieve all public figures mentioned in the text provided.
    :param text: text of the article
    :param api_key: API key. Only Groq supported so far.
    :param max_tokens: max tokens for the LLM call
    :param model: model adopted
    :return: the LLM call output with the list of persons.
    """
    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Sei un analista di notizie."},
            {
                "role": "user",
                "content": f"""
Sei un analista di notizie. Dal testo fornito, che è un articolo di notizie ottenuto tramite scraping di HTML,
devi individuare:
1. Tutti i nomi propri di personaggi pubblici menzionati.

Non aggiungere nessun altro commento o testo oltre all'oggetto JSON. Segui **rigorosamente** queste istruzioni.

Il risultato dovrà essere **esclusivamente** un oggetto JSON con la seguente struttura:

{{
  "persons": ["<string> (lista dei nomi propri di personaggi pubblici menzionati, se presenti, altrimenti lista vuota)"]
}}

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
        3
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
