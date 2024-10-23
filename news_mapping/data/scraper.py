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
    clean_with_genai: bool = True,
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

    if clean_with_genai:
        try:
            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a news article scraper and analyzer.",
                    },
                    {
                        "role": "user",
                        "content": f"""You are tasked with cleaning a string containing the title, author, and article of a scraped website.
                   Your goal is to:
                   1. Remove all irrelevant noise (ads, announcements, etc.).
                   2. Return a clean version of the article that only includes the title, author, and content.
                   3. Keep the format as: Title, Author, and Article Body.
                   Clean the following text: {text}
                """,
                    },
                ],
                model=model,
                max_tokens=max_tokens,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error during Groq API call: {e}")
            return None
    else:
        return text
