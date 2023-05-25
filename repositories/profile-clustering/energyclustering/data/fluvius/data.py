from pathlib import Path
import pandas as pd


ROOT_PATHS = [
    Path('/cw/dtaiproj/ml/2020-FLAIR-VITO/profile-clustering'),
    Path(__file__).parent/'data'/'profile-clustering',
    Path()/'data'
    # lola you can add your path here :D
]
ROOT_PATH = next((path for path in ROOT_PATHS if path.exists()), Path())
# take the first path that exists
PRE_PATH =  ROOT_PATH / 'preprocessed'/'combined'


class FluviusDataContainer:
    PATHS = [
        # for the web demo
        Path(__file__).parent/'data'/'distance_matrices',
        # used when Jonas mounts the data locally
        Path(__file__).parent/'data'/'profile-clustering'/'distance_matrices',
        # used when running remotely on DTAI infrastructure
        Path('/cw/dtaiproj/ml/2020-FLAIR-VITO/profile-clustering/distance_matrices'),
    ]
    def __init__(self, distance_matrix_name):
        path = next((path/distance_matrix_name for path in self.PATHS if (path/distance_matrix_name).exists()), Path())
        self.data_path = path
        self.info_df = None
        self.data_df = None
        self.distance_matrix = None

    def read_data(self):
        print('reading the data... this can take a while (if remotely mounted)')
        # self.info_df, data_df = read_data_pickle()
        # data_df = data_df.sort_index(ascending=False)
        # self.data_df = data_df.groupby('meterID').head(1)

        self.distance_matrix = pd.read_pickle(self.data_path/ 'full_distance_matrix.pkl')
        self.info_df = pd.read_pickle(self.data_path/'info.pkl')
        self.data_df = pd.read_pickle(self.data_path/'data.pkl').rename_axis(columns = 'timestamp')
        return self

    def get_profile_series(self, profile_id):
        return self.data_df.iloc[profile_id]


def read_data_pickle(include_incomplete_profiles = True, process_errors = True):
    info_path = PRE_PATH/'reindexed_info.pkl'
    if not include_incomplete_profiles:
        data_path = PRE_PATH/'reindexed_DST_data_subset_no_errors.pkl'
    else:
        if process_errors:
            data_path = PRE_PATH/'reindexed_DST_data_masked_errors.pkl'
        else:
            data_path = PRE_PATH/'reindexed_DST_data.pkl'
    info_df = pd.read_pickle(info_path)
    data_df = pd.read_pickle(data_path)
    return info_df, data_df
