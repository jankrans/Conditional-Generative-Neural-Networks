import matplotlib.pyplot as plt
import seaborn as sns

class ConsumptionClusteringInspector:
    def __init__(self, consumption_sampler, daily_data_df, training_set, test_set):
        self.sampler = consumption_sampler
        self.train_set = training_set
        self.test_set = test_set
        self.daily_data_df = daily_data_df.rename_axis(('meterID', 'day'), axis = 0).rename_axis('timestamp', axis = 1)

    @property
    def train_data(self):
        return self.daily_data_df.loc[self.train_set]

    @property
    def test_data(self):
        return self.daily_data_df.loc[self.test_data]

    @property
    def clustering(self):
        return self.sampler.clustering.rename_axis('meterID', axis = 0).rename('cluster_idx')


    def cluster_size_df(self):
        return self.sampler.clustering.value_counts()

    def plot_clusters_boxplot(self):
        plot_df = (
            self.train_data
                .stack()
                .to_frame('value')
                .join(self.clustering)
                .reset_index()
        )
        g = sns.FacetGrid(plot_df, row="cluster_idx", sharey=False, aspect=3)
        g.map(sns.boxplot, "timestamp", "value")

    def plot_clusters_lines(self):
        plot_df = (
            self.train_data
                .stack()
                .to_frame('value')
                .join(self.clustering)
                .reset_index()
        )
        g = sns.FacetGrid(plot_df, row="cluster_idx", hue='color', sharey=False, aspect=3)
        g.map(sns.lineplot, "timestamp", "value", size=0.1)

