import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances_chunked
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def calculate_energy_score(probs_for_sample, samples, correct_profile):
    """
        samples: 2d numpy array shape (#samples, #dim) , rows are predicted samples
        probs_for_sample: 1d numpy array shape (#samples), for each sample in samples the probability that it will get sampled
        correct_profile: 1d numpy array shape (#dim), the ground truth sample
    """
    def reduce_function(chunk, start):
        sums = (chunk * probs_for_sample).sum(axis = 1)
        return sums*probs_for_sample[start:start + sums.shape[0]]
    # because the full distance array gets very very large (GB's), process the chunks seperately into the needed sum
    second_term = 0
    for chunk in pairwise_distances_chunked(samples, reduce_func = reduce_function):
        second_term += chunk.sum(axis = None)
    distances_between_test_and_training_days = euclidean_distances(samples, correct_profile.reshape((1, -1))).squeeze()
    
    first_term = np.sum(probs_for_sample * distances_between_test_and_training_days)
    return first_term - 0.5*second_term
    
def get_data(year=None):

    if year == '2016':
        train = pd.read_pickle(r'C:\Thesis\data\train\train2016.pkl')
        val = pd.read_pickle(r'C:\Thesis\data\train\val2016.pkl')
        test = pd.read_pickle(r'C:\Thesis\data\train\test2016.pkl')
    elif year == '2021':
        train = pd.read_pickle(r'C:\Thesis\data\train\train2021.pkl')
        val = pd.read_pickle(r'C:\Thesis\data\train\val2021.pkl')
        test = pd.read_pickle(r'C:\Thesis\data\train\test2021.pkl')

    else:
        train = pd.concat([pd.read_pickle(r'C:\Thesis\data\train\train2016.pkl'),pd.read_pickle(r'C:\Thesis\data\train\train2021.pkl')])
        val = pd.concat([pd.read_pickle(r'C:\Thesis\data\train\val2016.pkl'),pd.read_pickle(r'C:\Thesis\data\train\val2021.pkl')])
        test = pd.concat([pd.read_pickle(r'C:\Thesis\data\train\test2016.pkl'),pd.read_pickle(r'C:\Thesis\data\train\test2021.pkl')])

    col_feat = train.columns[0:96]
    col_attr = train.columns[96:-1] #Dont include ID

    return train[ col_feat], train[ col_attr],val[ col_feat], val[ col_attr],test[ col_feat], test[ col_attr]

def standardize_data(train,val,test):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    return train,val,test, scaler

def one_hot_encode_columns(data, columns = ['day_of_week','season','PV']):

    for col in columns:
        onehot = pd.get_dummies(data[col], prefix=col)
        data = data.join(onehot)

    data.drop(columns=columns,inplace=True)
    return data

def minmax_scale_attributes(train,val,test):
     scaler = MinMaxScaler()
     train = scaler.fit_transform(train)
     val = scaler.transform(val)
     test = scaler.transform(test)

     return train,val,test,scaler

def get_similar_days(features, attributes, org_attribute, k = 50):
    distances = np.linalg.norm(attributes - org_attribute, axis=1)
    similar_idxs = np.argpartition(distances, k)[:k]
    return features[similar_idxs]

def calculate_average(features):
    features = features.reshape((len(features),-1))
    return  np.mean(features,axis=0)

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y
    
def get_random_days(dataset,n_samples=250, replace=False):
    return dataset[np.random.choice(dataset.shape[0], n_samples, replace=False)]