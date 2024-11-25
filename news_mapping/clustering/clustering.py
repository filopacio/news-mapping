from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from groq import Groq
import pandas as pd
import hdbscan
import numpy as np
from collections import Counter

from news_mapping.text_analysis.utils import (
    evaluate_string,
    extract_inside_braces)

from news_mapping.clustering.utils import replace_values_from_dict



def vectorize_topics(topics: list):
    """
    Convert a list of topics into vectors using Word2Vec.
    """
    # Train Word2Vec on topics
    word2vec_model = Word2Vec(sentences=[topic.split() for topic in topics], vector_size=100, window=5, min_count=1,
                              workers=4)

    # Vectorize each topic by averaging word vectors
    topic_vectors = np.array([word2vec_model.wv[topic.split()].mean(axis=0) for topic in topics])

    return topic_vectors


def kmeans_clustering(topic_vectors, num_clusters):
    """
    Perform K-means clustering on the vectorized topics.

    Args:
        topic_vectors: A numpy array of vectorized topics.
        num_clusters: Number of clusters to create.

    Returns:
        cluster_labels: The K-means cluster labels for each topic.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(topic_vectors)
    return cluster_labels


def hdbscan_clustering(topic_vectors):
    """
    Perform HDBSCAN clustering on the vectorized topics.

    Args:
        topic_vectors: A numpy array of vectorized topics.

    Returns:
        cluster_labels: The HDBSCAN cluster labels for each topic.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = clusterer.fit_predict(topic_vectors)
    return cluster_labels


def cluster_topics(dataframe: pd.DataFrame, topics: list = None):
    """
    Perform clustering on the topics using K-means if topics are provided, else HDBSCAN.
    When topics are provided, force them to become the centroids for clustering.

    Args:
        dataframe: A pandas dataframe containing the topics column.
        topics: A list of predefined topics to determine the number of clusters for K-means (optional).

    Returns:
        dataframe: The updated dataframe with clustered topics.
    """
    # Step 1: Vectorize the topics
    topic_list = dataframe['topics'].tolist()
    topic_vectors = np.array(vectorize_topics(topic_list))  # Assumed to be a function that returns vectorized topics

    # Step 2: Check if predefined topics are provided
    if topics:
        # Vectorize predefined topics
        predefined_vectors = np.array(vectorize_topics(topics))  # Vectorize predefined topics as well

        # Combine predefined topics with original topics
        all_vectors = np.concatenate((topic_vectors, predefined_vectors), axis=0)

        # Perform K-means clustering using predefined topics as initial centroids
        num_clusters = len(topics)
        kmeans = KMeans(n_clusters=num_clusters, init=predefined_vectors,
                        n_init=1)  # Use predefined topics as centroids
        all_labels = kmeans.fit_predict(all_vectors)

        # Extract labels for original topics only
        dataframe['topic_cluster'] = all_labels[:len(topic_list)]
    else:
        # Use HDBSCAN clustering if no predefined topics
        dataframe['topic_cluster'] = hdbscan_clustering(topic_vectors)

    # Step 3: Assign the predefined topics as the cluster representatives
    clustered_topics = []
    for cluster_label in sorted(dataframe['topic_cluster'].unique()):
        if topics:
            # Assign the predefined topic as the representative for each cluster
            representative_topic = topics[cluster_label]
        else:
            # If no topics provided, find the most common topic within the cluster
            cluster_topics = dataframe[dataframe['topic_cluster'] == cluster_label]['topics']
            representative_topic = Counter(cluster_topics).most_common(1)[0][0]

        clustered_topics.append(representative_topic)

    # Map cluster labels to representative topics
    cluster_to_topic = {label: topic for label, topic in
                        zip(sorted(dataframe['topic_cluster'].unique()), clustered_topics)}

    # Update 'topics' column with representative topics
    dataframe['topics'] = dataframe['topic_cluster'].map(cluster_to_topic)

    return dataframe

def cluster_topics_with_llm(
        dataframe: pd.DataFrame,
        api_key: str,
        model: str,
        topics: list = None,
):
    """
    """
    client = Groq(api_key=api_key)

    if topics:
        topics_string = f"""The labels must belong ABSOLUTELY to one of the following labels: {topics} """
    else:
        topics_string = ""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a news analyzer"},
            {
                "role": "user",
                "content": f"""
    You are a news analyzer with the task of clustering similar topics of news articles.
    You have a list of topics, cluster them to put very similar topics into the same cluster.

Here's the list:

{dataframe["topics"].unique()}
The output must be ONLY AND EXCLUSIVELY a json file where keys are the cluster label and the value is the list of topics 
clustered together.

    {topics_string}

    """,
            },
        ],
        model=model,
    )

    clusters = chat_completion.choices[0].message.content
    clusters = evaluate_string(extract_inside_braces(clusters))

    dataframe = replace_values_from_dict(dataframe, "topics", clusters)

    return dataframe