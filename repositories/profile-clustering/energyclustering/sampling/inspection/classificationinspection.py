import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from functools import partial
METRICS = dict(
    accuracy = accuracy_score,
    balanced_accuracy = balanced_accuracy_score,
)

PROB_METRICS = dict(
    log_loss = log_loss,
    roc_auc_ovr = partial(roc_auc_score, multi_class = 'ovr'),
    roc_auc_ovo = partial(roc_auc_score, multi_class = 'ovo')
)

class ClassificationInspection:
    def __init__(self, clusterer, classifier, data, info, training_set, test_set):
        # models
        self.clusterer = clusterer
        self.classifier = classifier

        # data
        self.data = data
        self.info = info

        # train and test set
        self.training_set = training_set
        self.test_set = test_set

        # desired result for the test set
        self.clustering = None

    def plot_tree(self):
        plt.figure(figsize = (30, 10))
        plot_tree(self.classifier, feature_names = self.info.columns)

    def plot_yearly_clustering_line(self, sample = None, col_wrap = None):
        plot_df = self.data.copy()
        clustering = self.clustering.rename_axis('meterID', axis = 0).to_frame('cluster_idx')
        if sample is not None:
            clustering = clustering.groupby('cluster_idx').apply(lambda x: x.sample(min(sample, len(x)))).droplevel(0, axis = 0).sort_index()
        plot_df = (
            plot_df
            .stack()
            .rename_axis(('meterID','timestamp'), axis = 0)
            .to_frame('value')
            .join(clustering, how = 'right')
            .reset_index()
            .assign(
                color = lambda x: x.meterID.astype('str')
            )

        )
        if col_wrap is not None:
            g = sns.FacetGrid(plot_df, col = 'cluster_idx', hue = 'color', sharey = False, aspect = 1, col_wrap = col_wrap)
        else:
            g = sns.FacetGrid(plot_df, row="cluster_idx", hue='color', sharey=False, aspect=9)
        g.map(sns.lineplot, "timestamp", "value", size=0.1)
        plt.show()

    def plot_clustering_line(self, sample = None, col_wrap = None):
        plot_df = self.data.copy()
        clustering = self.clustering.rename_axis(['meterID', 'day'], axis = 0).to_frame('cluster_idx')
        if sample is not None:
            clustering = clustering.groupby('cluster_idx').apply(lambda x: x.sample(min(sample, len(x)))).droplevel(0, axis = 0).sort_index()
        plot_df = (
            plot_df
            .stack()
            .rename_axis(('meterID','day','timestamp'), axis = 0)
            .to_frame('value')
            .join(clustering, how = 'right')
            .reset_index()
            .assign(
                color = lambda x: x.meterID.astype('str') + x.day.astype('str')
            )

        )
        if col_wrap is not None:
            g = sns.FacetGrid(plot_df, col = 'cluster_idx', hue = 'color', sharey = False, aspect = 1, col_wrap = col_wrap)
        else:
            g = sns.FacetGrid(plot_df, row="cluster_idx", hue='color', sharey=False, aspect=3)
        g.map(sns.lineplot, "timestamp", "value", size=0.1)
        plt.show()



    @property
    def training_data(self):
        return self.data.loc[self.training_set], self.info.loc[self.training_set]

    @property
    def test_data(self):
        return self.data.loc[self.test_set], self.info.loc[self.test_set]

    def fit_model(self):
        # cluster the dataset
        clustering = self.clusterer.old_fit(self.data).labels_
        self.clustering = pd.Series(clustering, index = self.data.index).rename('cluster_idx')

        # learn a classifier
        train_set_clustering = self.clustering.loc[self.training_set]
        train_info = self.info.loc[self.training_set]
        self.classifier.old_fit(train_info, train_set_clustering)
        return self

    def training_cluster_size_df(self):
        return self.clustering.loc[self.training_set].value_counts().to_frame('#items')

    def cluster_size_df(self):
        return self.clustering.value_counts().to_frame('#items')


    def confusion_matrix(self, sort_by_size = False, styled = True):
        test_info = self.info.loc[self.test_set]

        y_test = self.clustering.loc[self.test_set].to_numpy()
        y_pred_probs = self.classifier.predict_proba(test_info)

        all_clusters = np.sort(self.clustering.unique())



        confusion_matrix = (
            pd.DataFrame(y_pred_probs, index=y_test.astype('int'), columns = self.classifier.classes_)
                .groupby(level=0).sum()
                .apply(lambda x: x / x.sum(), axis=1)
                .rename_axis('True Cluster', axis=0)
                .rename_axis('Predicted Cluster', axis=1)
                .reindex(all_clusters, axis = 0)
                .reindex(all_clusters, axis = 1)
                .fillna(0)
        )

        if sort_by_size:
            cluster_sizes = self.cluster_size_df()

            # get non training clusters
            train_set_clustering = self.clustering.loc[self.training_set]
            train_set_clusters = train_set_clustering.unique()
            test_clusters = set(all_clusters)
            test_clusters.difference_update(train_set_clusters)

            sorted_train_clusters = cluster_sizes.loc[train_set_clusters].sort_values('#items', ascending = False).index.to_list()

            # training clusters sorted + test clusters at the end
            new_row_index = sorted_train_clusters + list(test_clusters)

            # only training clusters sorted
            new_column_index = sorted_train_clusters

            confusion_matrix = (confusion_matrix
                    .reindex(new_row_index, axis = 0)
                    .reindex(new_column_index, axis = 1)
            )

        if styled:
            confusion_matrix = (
                confusion_matrix.style
                    .background_gradient(axis=1)
                    .format({i: '{:.2f}' for i in confusion_matrix.columns})
            )

        return confusion_matrix

    def classification_performance(self):
        test_info = self.info.loc[self.test_set]

        y_test = self.clustering.loc[self.test_set].to_numpy()
        y_pred = self.classifier.predict(test_info)
        y_pred_probs = self.classifier.predict_proba(test_info)

        all_clusters = np.sort(self.clustering.unique())
        correct_y_pred_probs = np.zeros((y_pred_probs.shape[0], all_clusters.shape[0]))
        correct_y_pred_probs[:,self.classifier.classes_] = y_pred_probs

        scores = pd.Series(dtype='float')
        for metric, score_f in METRICS.items():
            scores[metric] = score_f(y_test, y_pred)
        for metric, score_f in PROB_METRICS.items():
            try:
                scores[metric] = score_f(y_test, correct_y_pred_probs, labels = all_clusters)
            except:
                scores[metric] = np.NaN
        return scores



