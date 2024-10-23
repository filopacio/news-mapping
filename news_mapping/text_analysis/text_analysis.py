import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from news_mapping.data.scraper import google_news_articles, scrape_url
from news_mapping.data.wrangler import (
    obtain_topics,
    obtain_persons,
    summarize_text,
)

from news_mapping.text_analysis.utils import (
    evaluate_string,
    extract_inside_braces,
)


class NewsProcess:
    """
    Based on some keywords (a query), a list of newspapers (sources) and optionally a list of
    topics to recognize (otherwise it will just find topics autonomously), this library scrapes
    articles from Google News, summarizes them, and extracts the main topics discussed,
    the person mentioned, and the newspapers discussing such topics and such persons in a certain
    time window, if specified (otherwise it will take as default last month).
    It leverages groq API (free usage of many LLMs) and serp API to access Google News (free up to 100 calls per month)
    """
    def __init__(
            self,
            query: str,
            serpapi_key: str,
            groq_api_key: str,
            sources: list,
            topics: list,
            start_date: str = (datetime.today() - relativedelta(months=1)).strftime('%Y-%m-%d'),
            end_date: str = datetime.today().strftime('%Y-%m-%d'),
            model: str = "mixtral-8x7b-32768"
    ):
        self.SERPAPI_KEY = serpapi_key
        self.GROQ_API_KEY = groq_api_key
        self.sources = sources
        self.topics = topics
        self.start_date = start_date
        self.end_date = end_date
        self.model = model
        self.query = query

    def scrape_articles(self):
        """
        Retrieve latest batch of articles from Google News, if scrape_new = True, then scrape. new batch to be
        added to the currently available batch.
        """
        dataframe = pd.DataFrame()

        for s in self.sources:
            df_t = google_news_articles(
                api_key=self.SERPAPI_KEY, keywords=f"{s} {self.query}"
            )
            dataframe = pd.concat([dataframe, df_t]).reset_index(drop=True)


        dataframe = dataframe.rename(columns={"source": "newspaper"})
        return dataframe[["title", "newspaper", "link", "date"]]

    def process_articles(self, dataframe: pd.DataFrame):
        """
        From scraped articles, summarize them, obtain topics and persons mentioned in the articles, and prepare
        output for relevant use.
        """

        dataframe["date"] = pd.to_datetime(
            dataframe["date"].apply(lambda x: x.split(",")[0]), format="%m/%d/%Y"
        )
        dataframe = dataframe[
            (dataframe["date"] >= pd.to_datetime(self.start_date))
            & (dataframe["date"] <= pd.to_datetime(self.end_date))
        ]
        dataframe["newspaper"] = dataframe["newspaper"].apply(lambda x: x["name"])
        dataframe["text"] = dataframe["link"].apply(
            lambda url: scrape_url(url=url, clean_with_genai=False)
        )
        dataframe = dataframe[
            dataframe["text"].astype(str).apply(len) < 15000
        ].reset_index(drop=True)
        dataframe["text_summary"] = dataframe["text"].apply(
            lambda text: summarize_text(
                text=text, api_key=self.GROQ_API_KEY, model=self.model
            )
        )
        dataframe["topics"] = dataframe["text_summary"].apply(
            lambda text: obtain_topics(
                text=text,
                api_key=self.GROQ_API_KEY,
                topics_to_scrape=self.topics,
                model=self.model,
            )
        )
        dataframe["topics"] = (
            dataframe["topics"].apply(extract_inside_braces).apply(evaluate_string)
        )

        dataframe["persons"] = dataframe["text_summary"].apply(
            lambda text: obtain_persons(
                text=text,
                api_key=self.GROQ_API_KEY,
                model=self.model,
            )
        )
        dataframe["persons"] = (
            dataframe["persons"].apply(extract_inside_braces).apply(evaluate_string)
        )
        dataframe = dataframe[(dataframe["topics"] != {}) & (dataframe["persons"] != {})]

        dataframe['topics'] = dataframe['topics'].apply(lambda x: x['topic'])
        dataframe['persons'] = dataframe['persons'].apply(lambda x: x['persons'])


        return dataframe
