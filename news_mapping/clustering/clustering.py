from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import hdbscan
import numpy as np
from collections import Counter
import pandas as pd


def vectorize_topics(topics: list):
    """
    Convert a list of topics into vectors using Word2Vec.
    """
    word2vec_model = Word2Vec(sentences=[topic.split() for topic in topics], vector_size=100, window=5, min_count=1,
                              workers=4)

    topic_vectors = np.array([word2vec_model.wv[topic.split()].mean(axis=0) for topic in topics])

    return topic_vectors


def kmeans_clustering(topic_vectors, **kwargs):
    """
    Perform K-means clustering on the vectorized topics.
    """
    kmeans = KMeans(**kwargs)
    cluster_labels = kmeans.fit_predict(topic_vectors)
    return cluster_labels


def hdbscan_clustering(topic_vectors, **kwargs):
    """
    Perform HDBSCAN clustering on the vectorized topics.
    """
    clusterer = hdbscan.HDBSCAN(**kwargs)
    cluster_labels = clusterer.fit_predict(topic_vectors)
    return cluster_labels


def cluster_topics(dataframe: pd.DataFrame, topics: list = None):
    """
    Perform clustering on the topics using K-means if topics are provided, else HDBSCAN.
    When topics are provided, force them to become the centroids for clustering.
    """
    topic_list = dataframe['topics'].tolist()
    topic_vectors = vectorize_topics(topic_list)

    if topics:
        # Vectorize predefined topics
        predefined_vectors = vectorize_topics(topics)

        # Combine predefined topics with original topics
        all_vectors = np.concatenate((topic_vectors, predefined_vectors), axis=0)

        # Perform K-means clustering using predefined topics as initial centroids
        n_clusters = len(topics)
        all_labels = kmeans_clustering(all_vectors, n_clusters=n_clusters, init=predefined_vectors, n_init=1)

        # Extract labels for original topics only
        dataframe['topic_cluster'] = all_labels[:len(topic_vectors)]
    else:
        # Use HDBSCAN clustering if no predefined topics
        dataframe['topic_cluster'] = hdbscan_clustering(topic_vectors)

    clustered_topics = []
    cluster_sizes = dataframe['topic_cluster'].value_counts()

    for cluster_label in sorted(dataframe['topic_cluster'].unique()):
        if cluster_sizes[cluster_label] < 2:
            representative_topic = "Outlier Topic"
        else:
            if topics:
                representative_topic = topics[cluster_label]
            else:
                cluster_topics = dataframe[dataframe['topic_cluster'] == cluster_label]['topics']
                representative_topic = Counter(cluster_topics).most_common(1)[0][0]

        clustered_topics.append(representative_topic)

    cluster_to_topic = {label: topic for label, topic in zip(sorted(dataframe['topic_cluster'].unique()), clustered_topics)}

    dataframe['topics'] = dataframe['topic_cluster'].map(cluster_to_topic)

    return dataframe

