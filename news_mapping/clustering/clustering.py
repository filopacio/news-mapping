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
    Then update the 'topics' column with the representative topic of each cluster.

    Args:
        dataframe: A pandas dataframe containing the topics column.
        topics: A list of predefined topics to determine the number of clusters for K-means (optional).

    Returns:
        dataframe: The updated dataframe with clustered topics.
    """
    # Get the list of topics from the dataframe
    topic_list = dataframe['topics'].tolist()

    # Step 1: Vectorize the topics
    topic_vectors = vectorize_topics(topic_list)

    # Step 2: Perform clustering based on predefined topics or not
    if topics is not None:
        # Use K-means clustering if predefined topics are provided
        num_clusters = len(topics) if len(topics) < len(topic_list) else len(topic_list)
        dataframe['topic_cluster'] = kmeans_clustering(topic_vectors, num_clusters)
    else:
        # Use HDBSCAN clustering if no predefined topics
        dataframe['topic_cluster'] = hdbscan_clustering(topic_vectors)

    # Step 3: Assign a representative topic to each cluster
    clustered_topics = []
    for cluster_label in sorted(dataframe['topic_cluster'].unique()):
        # Get all topics in the current cluster
        cluster_topics = dataframe[dataframe['topic_cluster'] == cluster_label]['topics']

        # Find the most common topic in this cluster
        most_common_topic = Counter(cluster_topics).most_common(1)[0][0]

        # Assign this topic as the representative topic for the cluster
        clustered_topics.append(most_common_topic)

    # Map cluster labels to representative topics
    cluster_to_topic = {label: topic for label, topic in
                        zip(sorted(dataframe['topic_cluster'].unique()), clustered_topics)}

    # Update 'topics' column with representative topics
    dataframe['topics'] = dataframe['topic_cluster'].map(cluster_to_topic)

    return dataframe
