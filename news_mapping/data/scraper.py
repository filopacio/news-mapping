import requests
import pandas as pd
from groq import Groq
from bs4 import BeautifulSoup
from serpapi.google_search import GoogleSearch


def google_news_articles(
    api_key: str, keywords: str, limit: int = 10000, country: str = "it"
) -> pd.DataFrame:
    """
    :param api_key: SerpAPI key
    :param keywords: keywords to query in Google News
    :param limit: max number of articles scraped
    :param country: desired country from where articles scraped are from
    :return: pandas dataframe with all articles obtained with SerpAPI
    """
    dataframe = pd.DataFrame()
    params = {
        "engine": "google_news",
        "q": keywords,
        "gl": country,
        "num": limit,
        "api_key": api_key,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    news_results = results["news_results"]
    dataframe_temp = pd.DataFrame(news_results)
    dataframe = pd.concat([dataframe, dataframe_temp]).reset_index(drop=True)
    return dataframe


def scrape_url(
    url: str,
    clean_with_llm: bool = False,
    max_tokens: int = 1024,
    model: str = "llama3-70b-8192",
    api_key: str = None,
) -> str:
    """
    Scraping function from URL link with Groq API (llama models for free without need of downloading them)
    As of now, only Groq API is supported.
    :param clean_with_genai: Boolean indicating whether to clean the text with Groq API.
    :param url: URL string of article to scrape.
    :param max_tokens: Maximum number of tokens for the output.
    :param model: Model to use. Default is llama 70b 8192.
    :param api_key: API key for Groq.
    :return: Corpus of article as a string, or None if an error occurs.
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print("Request not successful")
            return None
    except requests.RequestException as e:
        print(f"Error during request: {e}")
        return None

    try:
        html_string = response.text
        soup = BeautifulSoup(html_string, "html.parser")
        text = soup.get_text().replace("\n", "")
    except Exception as e:
        print(f"Error during HTML parsing: {e}")
        return None

    if clean_with_llm:
        try:
            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Sei un estrattore e analizzatore di articoli di giornale."
                    },
                    {
                        "role": "user",
                        "content": f"""Il tuo compito è pulire una stringa che contiene il titolo, l'autore e l'articolo 
                        di un sito web estratto. Il tuo obiettivo è:
                        1. Rimuovere tutto il rumore irrilevante (annunci, pubblicità, ecc.).
                        2. Restituire una versione pulita dell'articolo che includa solo il titolo, l'autore e il contenuto.
                        3. Mantieni il formato come: Titolo, Autore e Corpo dell'Articolo.
                        NON aggiungere nessun'altra parola di nessun tipo al tuo riassunto!! E' molto importante che 
                        segui attentamente queste istruzioni.
                        Pulisci il seguente testo: {text}
                        """
                    }
                    ,
                ],
                model=model,
                max_tokens=max_tokens,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error during Groq API call: {e}")
            return ""
    else:
        return text
