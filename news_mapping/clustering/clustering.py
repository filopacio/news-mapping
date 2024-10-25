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
    topic_list = dataframe['topics'].tolist()
    topic_vectors = vectorize_topics(topic_list)  # Assumed to be a function that returns vectorized topics

    if topics:
        predefined_vectors = vectorize_topics(topics)  # Vectorize predefined topics as well
        all_vectors = topic_vectors + predefined_vectors  # Combine predefined topics with original topics

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
            representative_topic = topics[cluster_label]
        else:
            cluster_topics = dataframe[dataframe['topic_cluster'] == cluster_label]['topics']
            representative_topic = Counter(cluster_topics).most_common(1)[0][0]

        clustered_topics.append(representative_topic)

    cluster_to_topic = {label: topic for label, topic in
                        zip(sorted(dataframe['topic_cluster'].unique()), clustered_topics)}

    dataframe['topics'] = dataframe['topic_cluster'].map(cluster_to_topic)

    return dataframe