import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

from news_mapping.data.scraper import google_news_articles, scrape_url
from news_mapping.data.wrangler import (
    get_newspaper_topics_persons,
  #  summarize_text,
)

from news_mapping.text_analysis.utils import (
    evaluate_string,
    extract_inside_braces,
    filter_newspapers,
    map_incomplete_to_full_names
)

from news_mapping.clustering.clustering import cluster_topics

tqdm.pandas()

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
            topics: list = None,
            start_date: str = (datetime.today() - relativedelta(months=1)).strftime('%Y-%m-%d'),
            end_date: str = datetime.today().strftime('%Y-%m-%d'),
            model: str = "mixtral-8x7b-32768",
            batch_size: int = 10
    ):
        self.SERPAPI_KEY = serpapi_key
        self.GROQ_API_KEY = groq_api_key
        self.sources = sources
        self.topics = topics
        self.start_date = start_date
        self.end_date = end_date
        self.model = model
        self.query = query
        self.batch_size = batch_size

    def scrape_articles(self) ->  pd.DataFrame:
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
        dataframe["date"] = pd.to_datetime(
            dataframe["date"].apply(lambda x: x.split(",")[0]), format="%m/%d/%Y"
        )
        dataframe = dataframe[
            (dataframe["date"] >= pd.to_datetime(self.start_date))
            & (dataframe["date"] <= pd.to_datetime(self.end_date))
            ]
        dataframe["newspaper"] = dataframe["newspaper"].apply(lambda x: x["name"])

        dataframe = filter_newspapers(dataframe, self.sources)

        dataframe = dataframe[["title", "newspaper", "link", "date"]]
        dataframe = dataframe.reset_index(drop=True)

        return dataframe


    def process_articles(self, dataframe: pd.DataFrame):
        """
        From scraped articles, summarize them, obtain topics and persons mentioned in the articles, and prepare
        output for relevant use.
        """

        print("Scraping And Cleaning URLs")
        dataframe["text"] = dataframe["link"].progress_apply(
            lambda url: scrape_url(url=url, api_key=self.GROQ_API_KEY, model=self.model, clean_with_genai=True)
        )

        dataframe = dataframe[
            dataframe["text"].astype(str).apply(len) < 15000
            ].reset_index(drop=True)

        print("Extracting Topics And Persons From Articles")
        dataframe = get_newspaper_topics_persons(dataframe=dataframe,
                                                 api_key=self.GROQ_API_KEY,
                                                 topics_to_scrape=self.topics,
                                                 model=self.model,
                                                 batch_size=self.batch_size
                                                 )

        dataframe = dataframe.dropna(subset="newspaper_topics_persons")

        # Separate 'topics' and 'persons' into their own columns
        dataframe["newspaper"] = dataframe["newspaper_topics_persons"].apply(lambda x: x["newspaper"])
        dataframe["topics"] = dataframe["newspaper_topics_persons"].apply(lambda x: x["topics"])
        dataframe["persons"] = dataframe["newspaper_topics_persons"].apply(lambda x: x["persons"])


        dataframe = dataframe[["title", "newspaper", "link", "date", "text", "topics", "persons"]]

        # Cluster topics to avoid extremely similar topics
        dataframe = cluster_topics(dataframe, self.topics)

        # Map duplicated names into single one (e.g. when only surname is given)
        dataframe = dataframe.explode("persons")
        dataframe["persons"] = dataframe["persons"].astype(str)
        dataframe["persons"] = dataframe["persons"].str.title()
        dataframe["persons"] = map_incomplete_to_full_names(dataframe["persons"])
        dataframe = dataframe.groupby(["title", "newspaper", "link",
                                       "date", "text", "topics"], as_index=False).agg({"persons": list})

        return dataframe
