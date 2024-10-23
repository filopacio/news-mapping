#  News Mapping European Elections

The projects maps in a multigraph structure topics, person and newspaper in the italian media coverage in the last 10 days before european elections 2024.

The process was the following:

- List of news articles by keyword(s) from Google News obtained using SerpAPI (100 API calls/month are free)
- URL scraping with requests and BeatifulSoup
- The corpus of the articles was cleaned with regex and LLM calls
- Topic Modelling performed with LLM calls
- Finding relationship between entities (nodes and edges) with LLM calls (relationships were defined a priori)
- Graph mapping with networkx

![Example](resources/graph_rep.png)
