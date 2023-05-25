from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt

from energyclustering.data.fluvius.data import FluviusDataContainer
from energyclustering.visualization.cluster_visualization import plot_pair

RESULT_DIR = Path(__file__).parent/'result'
from sklearn.metrics import adjusted_rand_score
import pickle

class COBRASResult:
    def __init__(self, result_dir, data_dir):
        assert (RESULT_DIR / result_dir).exists(), f"{result_dir} does not exist"
        assert data_dir.exists, f'{data_dir} does not exist'

        # name of the result
        self.name = result_dir
        self.path = RESULT_DIR / result_dir
        self.data_dir = data_dir

        # parse COBRAS results from disk
        self.queries = self.parse_queries(self.path/'queries.csv')
        self.clusterings = self.parse_clusterings(self.path/'clusters')
        self.super_instances = self.parse_superinstances(self.path/'superinstances')

        # load the data
        self._data_df = None
        self._info_df = None

    def get_clustering_df_of_representatives(self, cluster_index = -1):
        clustering = self.clusterings[cluster_index].astype('int')
        super_instances = self.super_instances[cluster_index]
        data_df = self.data_df.iloc[super_instances]
        labels = pd.Series(clustering[super_instances], index = data_df.index).to_frame('label')
        return labels, data_df

    def get_clustering_df(self, cluster_index=-1):
        """
            returns the data dataframe and a dataframe with the cluster indices of each profile
        """
        clustering = self.clusterings[cluster_index].astype('int')
        data_df = self.data_df
        return pd.Series(clustering, index = data_df.index).to_frame('label'), data_df

    @classmethod
    def parse_queries(cls, path):
        queries = pd.read_csv(path, header=None).rename(columns={0: 'i1', 1: 'i2', 2: 'isML'})
        queries['isML'] = (queries.isML == ' True').astype('bool')  # some weird parsing issue?
        return queries

    @classmethod
    def parse_clusterings(cls, path):
        cluster_paths = list(path.glob("clustering*.txt"))
        clusterings = None
        for cluster_path in cluster_paths:
            cluster_idx = int(cluster_path.stem[10:])
            clustering = np.loadtxt(str(cluster_path))
            if clusterings is None:
                nb_instances = len(clustering)
                nb_clusterings = len(cluster_paths)
                clusterings = np.zeros((nb_clusterings, nb_instances))
            clusterings[cluster_idx, :] = clustering
        return clusterings

    @classmethod
    def parse_superinstances(cls, path):
        cluster_paths = sorted(list((path).glob("superinstances*.txt")),
                               key=lambda x: int(x.stem[14:]))
        super_instances = []
        for cluster_path in cluster_paths:
            clustering = np.loadtxt(str(cluster_path)).astype('int')
            super_instances.append(clustering)
        return super_instances

    def get_queries_w_profile_ids(self):
        queries = self.queries.assign(
            i1 = lambda x: self.data_df.index[x.i1],
            i2 = lambda x: self.data_df.index[x.i2]
        )
        return queries

    @property
    def data_df(self):
        if self._data_df is None:
            self._read_data()
        return self._data_df

    @property
    def info_df(self):
        if self._info_df is None:
            self._read_data()
        return self._info_df

    def _read_data(self):
        data_container = FluviusDataContainer(self.data_dir).read_data()
        self._data_df = data_container.data_df
        self._info_df = data_container.info_df
        self._distance_matrix = data_container.distance_matrix

    @property
    def query_array(self):
        return self.queries[['i1', 'i2']].to_numpy()


class ResultInspector:
    def __init__(self, cobras_result, distance_matrix, name = None):
        self.name = name

        self.cobras_result = cobras_result
        self.distance_matrix = distance_matrix

        # to cache the queries with distances
        self._queries_with_distances = None

    @classmethod
    def from_path(cls, cobras_result, path, name = None):
        if name is None:
            name = path.stem
        with (path/'full_distance_matrix.pkl').open(mode='rb') as file:
            distance_matrix = pickle.load(file)
        return ResultInspector(cobras_result, distance_matrix, name)

    def rank_correlation_between_distances_and_queries(self):
        df = self.queries_with_distances
        return df.isML.replace({True: 0, False: 1}).corr(df.distance, method='spearman')

    def similarity_metric_histogram_chart(self, bins=50, filter=None):
        df = self.queries_with_distances
        if filter is not None:
            df = df.query(filter)
        return alt.Chart(df).transform_bin(
            field='distance',
            as_='binned_distance',
            bin=alt.Bin(maxbins=bins)
        ).mark_bar(opacity=0.5).encode(
            x='binned_distance:O',
            y='count():Q',
            row='isML:N',
            color='isML:N'
        )

    def plot_all_constraint_pairs_w_distances(self, type='all_days'):
        df = self.queries_with_distances.copy()
        data_df = self.cobras_result.data_df
        df.loc[:, ['i1', 'i2']] = df.loc[:, ['i1', 'i2']].applymap(lambda x: data_df.index[x])
        for idx, (i1, i2, is_ML, distance) in df.iterrows():
            self.plot_constraint_pair(i1, i2, is_ML, distance, type)

    def plot_constraint_pairs(self, nb=5, constraints=None, sort=None, type='all_days'):
        df = self.queries_with_distances.copy()
        data_df = self.cobras_result.data_df
        df.loc[:, ['i1', 'i2']] = df.loc[:, ['i1', 'i2']].applymap(lambda x: data_df.index[x])
        if constraints is not None:
            if constraints == 'ML':
                df = df[df.isML]
            else:
                df = df[~df.isML]
        if sort is not None:
            if 'asc' in sort.lower():
                df = df.sort_values('distance', ascending=True)
            else:
                df = df.sort_values('distance', ascending=False)
        df = df.iloc[0:nb]

        for idx, (i1, i2, is_ML, distance) in df.iterrows():
            self.plot_constraint_pair(i1, i2, is_ML, distance, type)

    def plot_constraint_pair(self, i1, i2, is_ML, distance, type):
        chart = plot_pair(i1, i2, self.cobras_result.data_df, type)
        constraint = 'ML' if is_ML else 'CL'
        chart.properties(title=f'{constraint}, distance= {distance}').resolve_scale(y='shared').display(renderer='png')

    def similarity_metric_distribution_chart(self, bandwidth=10, minsteps=100):
        df = self.queries_with_distances
        return alt.Chart(df).transform_density(
            'distance',
            groupby=['isML'],
            as_=['distance', 'density'],
            minsteps=minsteps,
            bandwidth=bandwidth,
        ).mark_area(opacity=0.5).encode(
            x='distance:Q',
            y='density:Q',
            color='isML:N'
        )

    @property
    def queries_with_distances(self):
        if self._queries_with_distances is None:
            # add distance between query points to each query
            def get_distance_between_query_points(row):
                i1, i2, _ = row
                return self.distance_matrix.loc[i1, i2]

            distances = self.cobras_result.get_queries_w_profile_ids().apply(get_distance_between_query_points, axis=1)
            self._queries_with_distances = self.cobras_result.queries.copy()
            self._queries_with_distances['distance'] = distances
        return self._queries_with_distances











class ResultComparison:
    def __init__(self, result1, result2):
        self.result1 = result1
        self.result2 = result2

    def clustering_similarity_df(self):
        clusterings1 = self.result1.clusterings
        clusterings2 = self.result2.clusterings
        similarities = [adjusted_rand_score(cluster1, cluster2) for cluster1, cluster2 in zip(clusterings1, clusterings2)]
        return pd.Series(similarities, name = 'similarity').rename_axis(index = 'clustering_idx').to_frame().reset_index()

    def superinstance_similarity_df(self):
        clusterings1 = self.result1.clusterings
        clusterings2 = self.result2.clusterings
        supers1 = self.result1.super_instances
        supers2 = self.result2.super_instances
        # common super-instances should be equal
        assert all((super1.sort() == super2.sort()) for super1, super2 in zip(supers1, supers2))
        similarities = [adjusted_rand_score(cluster1[supers], cluster2[supers]) for cluster1, cluster2, supers in zip(clusterings1, clusterings2, supers1)]
        return pd.Series(similarities, name='similarity').rename_axis(index='clustering_idx').to_frame().reset_index()

    def superinstance_similarity_chart(self):
        df = self.superinstance_similarity_df()
        return alt.Chart(df).mark_line().encode(
            x = 'clustering_idx:O',
            y = alt.Y('similarity:Q', title = 'ARI similarity')
        )

    def clustering_similarity_chart(self):
        df = self.clustering_similarity_df()
        return alt.Chart(df).mark_line().encode(
            x = 'clustering_idx:O',
            y = alt.Y('similarity:Q', title = 'ARI similarity')
        )