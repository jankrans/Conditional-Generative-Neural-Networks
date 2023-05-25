import altair as alt
from kde_diffusion import kde1d # pip install kde_diffusion
from sklearn.neighbors import KernelDensity # pip install scikit-learn
from scipy.stats import norm
import numpy as np
import pandas as pd
class NormalDistribution: 
    """
        Fits a normal distribution on the measurements
        A point is weird if it falls outside the 'threshold' likelihood interval 
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.mu = None
        self.std = None
        self.min_normal_value = None
        self.max_normal_value = None
    
    def fit(self, X):
        self.mu, self.std = norm.old_fit(X)
        self.min_normal_value, self.max_normal_value = norm.interval(self.threshold, self.mu, self.std)
    
    def test_value(self,value): 
        return value < self.min_normal_value or value > self.max_normal_value

class AutoKDEDistribution:
    """
        Fits a kernel density estimation on the measurements 
        To do this the bandwidths get determined automatically 
        However, it seems the bandwidth is chosen too small, so you can also use KDEDistribution and select the bandwidth yourself
        
        A value is weird if it is not among the threshold most likely values 
        TODO The previous statement is not really true, it is smoothed a bit but we might remove this later!
    """
    def __init__(self, threshold): 
        self.threshold = threshold
        self._nb_bins = 1024
        self.range = None
        self._bins = None
        self._bandwidth = None
        self._dens = None
        self.squash_threshold = None
        self._squashed_dens = None
        
    def get_vis_df(self): 
        df = pd.DataFrame()
        df['X']= self._bins + self.value_offset
        df['density'] = self._dens
        df['squashed_density'] = self._squashed_dens
        return df 
    
    def fit(self, data):

        # center the data around zero!
        mid = (np.max(data) - np.min(data)) / 2
        self.value_offset = mid
        data = data - mid
        
        # to be sure the DE can get to 0 
        min_value = np.min(data) - 2
        max_value = np.max(data) + 2
        self.range = min_value, max_value

        # do the density estimation
        auto_density, grid, auto_bandwidth = kde1d(
            data, self._nb_bins, limits=(min_value, max_value)
        )

        self._bandwidth = auto_bandwidth
        self._bins = grid
        self._dens = auto_density

        self.squash_threshold = self.calculate_squash_threshold()
        self._squashed_dens = np.clip(self._dens / self.squash_threshold, 0, 1)
    
    def test_value(self,value): 
        return self.score_samples(value) < 0.5
    
    def score_samples(self, X):
        X = X - self.value_offset
        # Can add linear interpolation here but that is probably unnecessary
        inds = np.digitize(X, self._bins)
        return self._squashed_dens[inds - 1]
    
    def calculate_squash_threshold(self):
        sort_indices = np.argsort(self._dens, )
        normalized_density = self._dens / np.sum(self._dens)
        sum = 0
        # biggest density index to smallest index
        for i in np.flip(sort_indices):
            sum += normalized_density[i]
            if sum >= self.threshold:
                break
        return self._dens[i]

class KDEDistribution:
    """
        Fits a kernel density estimation on the measurements 
        The bandwidth is a parameter of this class
        
        A value is weird if it is not among the threshold most likely values 
        TODO The previous statement is not really true, it is smoothed a bit but we might remove this later!
    """
    def __init__(self, threshold, bandwidth = 0.3): 
        self.threshold = threshold
        self._nb_bins = 1024
        self.range = None
        self._bins = None
        self._bandwidth = bandwidth
        self._dens = None
        self.squash_threshold = None
        self._squashed_dens = None
    
    def get_chart(self): 
        temp_df = self.get_vis_df()
        return alt.layer(
            alt.Chart(temp_df).mark_line().encode(
            x = 'X',
            y= 'density', 
        ),alt.Chart(temp_df).mark_line().encode(
            x = 'X',
            y= 'squashed_density', 
            color = alt.ColorValue('orange')
        )).interactive().properties(title = 'blue: KDE, orange: squashed KDE')
    def get_vis_df(self): 
        df = pd.DataFrame()
        df['X']= self._bins
        df['density'] = self._dens
        df['squashed_density'] = self._squashed_dens
        return df 
    
    def fit(self, data):
        # to be sure the DE can get to 0 
        min_value = np.min(data) - 2
        max_value = np.max(data) + 2
        self.range = min_value, max_value
        
        model = KernelDensity(kernel='gaussian', bandwidth=self._bandwidth)
        model.fit(data.reshape((-1,1)))
        self._bins = np.linspace(min_value, max_value, self._nb_bins)
        
        self._dens = model.score_samples(np.reshape(self._bins, (-1, 1)))
        self._dens = np.exp(self._dens).reshape((-1,))
        self.squash_threshold = self.calculate_squash_threshold()
        self._squashed_dens = np.clip(self._dens / self.squash_threshold, 0, 1)
    
    def test_value(self,value): 
        return self.score_samples(value) < 0.5
    
    def score_samples(self, X):
        X = X 
        # Can add linear interpolation here but that is probably unnecessary
        inds = np.digitize(X, self._bins)
        return self._squashed_dens[inds - 1]
    
    def calculate_squash_threshold(self):
        sort_indices = np.argsort(self._dens, )
        normalized_density = self._dens / np.sum(self._dens)
        sum = 0
        # biggest density index to smallest index
        for i in np.flip(sort_indices):
            sum += normalized_density[i]
            if sum >= self.threshold:
                break
        return self._dens[i]