import pandas as pd 
import numpy as np

svd_cache = dict()

def distance_scaled_principle_directions(row1, row2, components_to_use = None): 
    data_df = pd.DataFrame(np.array([row1,row2]), columns = pd.date_range('2016-1-1 00:00', '2016-12-31 23:45', freq = '15min'))
    day_df = to_daily_dataframe(data_df)
    svd1 = get_svd(day_df.loc[0].to_numpy(dtype = 'float'))
    svd2 = get_svd(day_df.loc[1].to_numpy(dtype = 'float'))
    return _distance_scaled_principle_directions(svd1, svd2, components_to_use)


def _distance_scaled_principle_directions(svd1, svd2, components_to_use = None):
    u1, s1, w1 = svd1
    u2, s2, w2 = svd2
    
    if components_to_use is None: 
        nb_of_components = s1.shape[0]
    else: 
        nb_of_components = components_to_use
        
    # scale the principle directions 
    r1 = np.diag(s1).dot(w1.T)[:nb_of_components]
    r2 = np.diag(s2).dot(w2.T)[:nb_of_components]
    return np.trace(np.linalg.multi_dot([r1.T, r2, r2.T, r1]))/nb_of_components



def to_daily_dataframe(df): 
    return  ( 
        df 
        # give index and columns a name
        .rename_axis(index = 'meterID', columns ='timestamp')
        # go to long format 
        .stack().to_frame('value')
        # reset the index
        .reset_index()
        # add columns for dayofyear and timeofday
        .assign(day = lambda x: x.timestamp.dt.dayofyear)
        .assign(time = lambda x: x.timestamp.dt.time)
        # add date to time for altair plotting 
        .assign(time = lambda x: pd.to_datetime(x.time, format = '%H:%M:%S'))
        # make nice pivot_table
        .pipe(lambda x: pd.pivot_table(x, index = ['meterID', 'day'], columns = 'time', values = 'value'))
        # drop the days that contain NaN's 
        .dropna(axis = 0) #TODO also drop days with data problems (zeros)
    )

def get_svd(data):
    data_tuple = tuple(map(tuple, data))
    if data_tuple in svd_cache: 
        return svd_cache[data_tuple]
    
    # do the svd 
    u, s, w = np.linalg.svd(data) # x = u diag(s) w   (w is already in the transposed form)
    
    # correct the signs of the principal components
    # when the dot product between a direction and [1, 1, ..., 1] is negative, change its sign.
    one_vector = np.ones(w.shape[0])
    for j in range(w.shape[1]): 
        if np.dot(w[j,:], one_vector)<0: 
            # These two operations do not break the consistency of the SVD multiplication. 
            # The modified matrices are still the SVD of x because of the almost-uniqueness of SVD.
             # The resulting similarity also does not change. 
            u[:,j] = -u[:,j] 
            w[j,:] = -w[j,:] 
    svd_cache[data_tuple] = u,s ,w
    return u, s, w

    