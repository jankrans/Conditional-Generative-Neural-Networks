import numpy as np
import ot
from energyclustering.util import add_date


def to_2d_numpy_array(profile, resample=4):
    daily_array = profile.reshape((-1, 96))
    chunks = np.split(daily_array, 96 / resample, axis=1)
    chunks = [chunk.sum(axis=1) for chunk in chunks]
    return np.stack(chunks, axis=1)


def wasserstein_distance_between_years(profile1, profile2):
    resample = 4
    days1 = to_2d_numpy_array(profile1.copy(), resample)
    days2 = to_2d_numpy_array(profile2.copy(), resample)

    total_distance = 0
    for column_idx in range(days1.shape[1]):
        p1_samples = days1[:, column_idx]
        p1_samples = p1_samples[~np.isnan(p1_samples)]
        p2_samples = days2[:, column_idx]
        p2_samples = p2_samples[~np.isnan(p2_samples)]
        total_distance += ot.emd2_1d(p1_samples, p2_samples)
    return total_distance


class WassersteinDistanceMeasure:
    def __init__(self, resample='1H'):
        self.resample = resample

    @staticmethod
    def profile_to_daily_df(profile, resample=None):
        daily_df = (
            profile.to_frame('value')
                .assign(
                time=lambda x: add_date(x.index.time),
                date=lambda x: x.index.date.astype('str')
            )
                .pivot_table(index='date', columns='time', values='value')
        )
        if resample is not None:
            daily_df = daily_df.resample(resample, axis=1).sum()
        return daily_df

    def distance(self, profile1, profile2):
        daily_df1 = self.profile_to_daily_df(profile1, self.resample)
        daily_df2 = self.profile_to_daily_df(profile2, self.resample)
        total_distance = 0
        for column in daily_df1:
            p1_samples = daily_df1[column].dropna()
            p2_samples = daily_df2[column].dropna()
            distance = ot.emd2_1d(p1_samples, p2_samples)
            total_distance += distance
        return total_distance
