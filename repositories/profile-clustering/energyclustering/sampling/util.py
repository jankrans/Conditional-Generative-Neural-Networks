import numpy as np
def cluster_probabilities_to_sample_probabilities(cluster_probabilities, clustering):
    cluster_idx, counts = np.unique(clustering, return_counts=True)

    # sort both arrays based on cluster_idx
    sort_idxs = np.argsort(cluster_idx)
    cluster_idx, counts = cluster_idx[sort_idxs], counts[sort_idxs]

    # sample_prob_per_cluster[i] = probability that a random sample from cluster i gets sampled
    sample_prob_per_cluster = 1 / counts
    # for each sample in the dataset, P(sample selected| cluster of sample is selected)
    p_sample_selected_from_cluster = sample_prob_per_cluster[clustering]
    p_cluster_of_sample_selected = cluster_probabilities[:, clustering]
    p_sample_selected = p_sample_selected_from_cluster * p_cluster_of_sample_selected

    return p_sample_selected