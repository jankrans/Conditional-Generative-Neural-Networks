import numpy as np 
import pandas as pd 
import itertools
from dtaidistance import dtw
# from dask.distributed import progress
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import time

def get_scaled_components(profile): 
    day_df = get_day_df(profile)
    components_df, representation_df = get_NMF(day_df, 10)
    scaled_components_df, times_used = scale_components_discrete(representation_df, components_df)
    return scaled_components_df, times_used

def get_distance_matrix_local(data_df): 
    scaled_component = list(map(get_scaled_components, [component for _, component in data_df.iterrows()]))
    tuples = itertools.product(range(0, data_df.shape[0]), range(0, data_df.shape[0]))
    distances = []
    for i,j in tuples: 
        start_time = time.time()
        print(i,j, end = '')
        distance = get_scaled_component_similarity_wrapper([scaled_component[i], scaled_component[j]])
        print(f'took {time.time() -start_time} seconds')
        distances.append(distance)
    return distances


def get_distance_matrix(data_df,client): 
    """
        calculates the distance between each pair of profiles in the data_df
    """
    scaled_component_futures = client.map(get_scaled_components, [component for _, component in data_df.iterrows()])
    distance_futures = client.map(get_scaled_component_similarity_wrapper, list(itertools.product(scaled_component_futures, scaled_component_futures)))
    progress(distance_futures, notebook = False, console = True)
    distances = client.gather(distance_futures)
    distance_matrix = (
        # make the series
        pd.Series(
            distances,
            index = pd.MultiIndex.from_product(
                [list(range(0, data_df.shape[0])),
                list(range(0, data_df.shape[0]))]
            )
        )
        # add names to index levels
        .rename_axis(index = ('from', 'to'))
        # make into frame
        .to_frame('distance')
        # make into matrix form
        .reset_index()
        .pipe(lambda x: pd.pivot_table(x, index = 'from', columns = 'to', values = 'distance'))
        
    )
    return distance_matrix
   
    
    
    
def get_component_similarity(profile1, profile2):
    # represent as day_df
    day_df1 = get_day_df(profile1)
    day_df2 = get_day_df(profile2)
    
    # do non negative matrix factorization
    components_df1, representation_df1 = get_NMF(day_df1, 10)
    components_df2, representation_df2 = get_NMF(day_df2, 10)
    
    # scale the components
#     scaled_components_df1, times_used1 = scale_components_simple(representation_df1, components_df1)
#     scaled_components_df2, times_used2 = scale_components_simple(representation_df2, components_df2)

    scaled_components_df1, times_used1 = scale_components_discrete(representation_df1, components_df1)
    scaled_components_df2, times_used2 = scale_components_discrete(representation_df2, components_df2)
    
    # match the scaled components and calculate the distance
    return get_scaled_component_similarity(scaled_components_df1, times_used1, scaled_components_df2, times_used2)

def add_date(series): 
    return pd.to_datetime(series, format='%H:%M:%S', exact = False)

def get_day_df(profile): 
    day_matrix = (
            profile
            .to_frame('value')
            # add time and date column
            .assign(
                time = lambda x: add_date(x.index.time), 
                date = lambda x: x.index.date.astype('str')
            )
            # make every day a row
            .pipe(lambda x: pd.pivot_table(x, index = 'date', columns = 'time', values = 'value'))
            # drop days that contain a NaN
            .dropna(axis = 0)
        )
    return day_matrix

def get_NMF(day_df, nb_of_components): 
    matrix = day_df.to_numpy()
    decomposer = NMF(10, max_iter = 100000, alpha = 0.1, l1_ratio = 1, init = 'nndsvd', random_state = 1234).fit(matrix)
    components = decomposer.components_
    components_df = (
        pd.DataFrame(components, columns = day_df.columns)
        .rename_axis(index = 'component_nb')
    )
    representation_matrix = decomposer.transform(matrix)
    representation_df = pd.DataFrame(representation_matrix, index = day_df.index).sort_index()
    return components_df, representation_df


def scale_single_component(coefficients, component): 
    """
        Unused this overcomplicated things! 
    """
    def get_new_coefficients(labels, centers): 
        new_coefficients = np.zeros(labels.shape)
        for label in np.unique(labels): 
            new_coefficients[labels == label] = centers[int(label)]
        return new_coefficients 
    
    non_zero_coefficients = coefficients[coefficients > 0]
    original_consumption = np.sum(component)*np.sum(non_zero_coefficients)
    
    # initial just use the mean coefficient
    n_clusters = 1
    centers = [np.mean(non_zero_coefficients)]
    labels = np.zeros(non_zero_coefficients.shape)
    new_coefficients = get_new_coefficients(labels, centers)
    reconstructed_consumption = np.sum(component)*np.sum(new_coefficients)
    reconstruction_error = abs(reconstructed_consumption- original_consumption)/original_consumption
    
    # iterate until reconstruction error is small 
    while reconstruction_error > 0.01 and n_clusters < 5:
        n_clusters += 1
        clusterer = KMeans(n_clusters)
        clusterer.fit(non_zero_coefficients.reshape((-1,1)))
        labels = clusterer.labels_
        centers = clusterer.cluster_centroids_
        new_coefficients = get_new_coefficients(labels, centers)
        reconstructed_consumption = np.sum(component)*np.sum(new_coefficients)
        reconstruction_error = abs(reconstructed_consumption - original_consumption)/original_consumption
    print(reconstruction_error)
    values, counts = np.unique(new_coefficients, return_counts = True)
    scaled_components = []
    for value in values: 
        scaled_components.append(component*value)
    return scaled_components, counts
    
    
def scale_components_discrete(representation_df, components_df, THRESHOLD =0.05): 
    new_coefficients = (
        representation_df
        .rename_axis(columns = 'component_nb')
        .stack()
        .to_frame('value')
        .pipe(lambda x: x[x.value!=0])
        .reset_index()
        .groupby('component_nb')
        ['value']
        .mean()
    )
    scaled_components = components_df.multiply(new_coefficients, axis = 0)
    times_used = (representation_df > 0).sum(axis = 0)
     # ensure that components that are never used are ignored
    times_used = times_used[times_used > 0]
    scaled_components = scaled_components.loc[times_used.index]
    return scaled_components, times_used
    
def scale_components_simple(representation_df, components_df, THRESHOLD=0.03): 
    """
        Scales the components in component_df by the most common coefficient that appears in the representation_df 
        A component is used in a day if the corresponding coefficient of that day is larger than THRESHOLD
        
        return: 
        - the scaled components
        - times_used per component the number the component was used with a coefficient > THRESHOLD
        
        note: at this point pretty simple, can be updated later to use the local maxima of the KDE instead of only the most common value
    """
    dfs = []
    for component_nb in components_df.index:
        component_values = representation_df[component_nb].pipe(lambda x: x[x>THRESHOLD])
        x, y = TreeKDE(bw = 0.01).old_fit(component_values.values).evaluate()
        kde_df = (
            pd.DataFrame()
            .assign(
                x = x, 
                y = y, 
                component_nb = component_nb
            )
        )
        dfs.append(kde_df)

    all_kde_dfs = pd.concat(dfs, axis = 0)
    most_common_coefficients = all_kde_dfs.groupby('component_nb')[['y', 'x']].max()['x']
    scaled_components = components_df.multiply(most_common_coefficients, axis = 0)
    times_used = (representation_df > THRESHOLD).sum(axis = 0)
    # ensure that components that are never used are ignored
    times_used = times_used[times_used > 0]
    scaled_components = scaled_components.loc[times_used.index]
    return scaled_components, times_used

def get_scaled_component_similarity_wrapper(argument):
    tuple1, tuple2 = argument
    return get_scaled_component_similarity(*tuple1, *tuple2)
    
def get_scaled_component_similarity(scaled_components_df1, times_used1, scaled_components_df2, times_used2): 
    component_matcher = ComponentMatcher(scaled_components_df1, times_used1, scaled_components_df2, times_used2)
    distance = 0
    # while there each component set has at least one component
    while not component_matcher.one_set_empty(): 
        # find the best match between components
        (comp1_idx, component1, used1), (comp2_idx, component2, used2), warping_path = component_matcher.get_best_aligned_pair()
#         print(f'matching {comp1_idx} with {comp2_idx} used {used1}, {used2}')
        # find the maximum amount of times_used that can be matched between the two components
        if used1 > used2: 
            diff = used1-used2
            # add diff*component1 to set 1
            component_matcher.change_times_used_set1(comp1_idx, diff)
            component_matcher.remove_component_from_set2(comp2_idx)
#             print(f'set2: remove {comp2_idx}')
        elif used2 > used1: 
            diff = used2-used1
            # add diff*component2 to set 2
            component_matcher.change_times_used_set2(comp2_idx, diff)
            component_matcher.remove_component_from_set1(comp1_idx)
#             print(f'set1: remove {comp1_idx}')
        else: 
            component_matcher.remove_component_from_set1(comp1_idx)
            component_matcher.remove_component_from_set2(comp2_idx)
#             print(f'set1: remove {comp1_idx}')
#             print(f'set2: remove {comp2_idx}')
        
        # amount of used that we can match
        used = min(used1, used2)
        
        # calculate the difference between the components
        difference = component1 - component2
        positive_sum = np.sum(difference[difference>0])
        negative_sum = -np.sum(difference[difference<0])
        if positive_sum > negative_sum: 
            # comp1 is the one to use 
            new_comp1 = difference
            new_comp1[new_comp1 < 0] = 0
            # add new comp1 with times_used used 
            component_matcher.add_component_to_set1(new_comp1, used)
#             print(f'set1: add {component_matcher.current_component_nb1}')
            distance += negative_sum*used
        else: 
            new_comp2 = -difference
            new_comp2[new_comp2 < 0] = 0
            # add new comp2 with times_used used 
            component_matcher.add_component_to_set2(new_comp2, used)
#             print(f'set2: add {component_matcher.current_component_nb2}')
            distance += positive_sum *used
#         print()
    left_over = component_matcher.get_non_empty_component_set()
#     if len(component_matcher.current_components1)==0:
#         print(f'left over set2: {component_matcher.current_components2}')
#     else: 
#         print(f'left over set1: {component_matcher.current_components1}')
    for times_used, component in left_over:   
        distance += times_used* np.sum(component)
    return distance 

class ComponentMatcher:
    """
        A class to help keep track of all the necessary information to make the matching algorithm pretty simple
        
        note: atm this might not be the best representation but it works which is most important for now ;) 
    """
    window = 4
    
    def __init__(self,scaled_components_df1, times_used1, scaled_components_df2, times_used2): 
        # keeps track of the times used attribute of each component
        self.times_used_dict = dict()
        # keeps track of all the aligned sequences
        self.aligned_sequence_dict = dict()
        # keeps track of all the dtw distances
        self.dtw_distances_dict = dict()
        
        # keeps track of the original components 
        self.original_components1 = {key: value.to_numpy() for key, value in scaled_components_df1.iterrows()}
        self.original_components2 = {key: value.to_numpy() for key, value in scaled_components_df2.iterrows()}
        
        # integers to use for the next components 
        self.current_component_nb1 = scaled_components_df1.index.max()
        self.current_component_nb2 = scaled_components_df2.index.max()
        
        # all components in set1
        self.current_components1 = set(scaled_components_df1.index)
        self.current_components2 = set(scaled_components_df2.index)
        
        # initialise the aligned sequence dict and dtw distances dict
        for comp_nb1, comp_nb2 in itertools.product(scaled_components_df1.index, scaled_components_df2.index):
            component1 = scaled_components_df1.loc[comp_nb1].to_numpy()
            component2 = scaled_components_df2.loc[comp_nb2].to_numpy()
            # added penalty to ensure no warping is preffered over warping 
            if self.window > 0:
                aligned_component1, best_path = dtw.warp(component1, component2, window = self.window, penalty = 0.1)
            else: 
                aligned_component1, best_path = component1, None
            dist = dtw.distance(component1, component2, window = self.window, penalty = 0.1)
            self.aligned_sequence_dict[(comp_nb1, comp_nb2)] = (aligned_component1, component2, best_path)
            self.dtw_distances_dict[(comp_nb1, comp_nb2)] = dist
        
        # initialise the times_used_dict 
        for index, value in times_used1.iteritems(): 
            self.times_used_dict[(1,index)] = value 
        for index, value in times_used2.iteritems(): 
            self.times_used_dict[(2,index)] = value 
    
    def one_set_empty(self): 
        return len(self.current_components1)==0 or len(self.current_components2)==0
    
    def get_non_empty_component_set(self): 
        assert self.one_set_empty()
        if len(self.current_components1) == 0: 
            return [(self.times_used_dict[(2, idx)], self.original_components2[idx]) for idx in self.current_components2]
        return [(self.times_used_dict[(1, idx)],self.original_components1[idx]) for idx in self.current_components1]
        
    def get_best_aligned_pair(self): 
        best_pair = min(self.dtw_distances_dict, key = self.dtw_distances_dict.get)
        component1, component2, best_path = self.aligned_sequence_dict[best_pair]
        times_used1 = self.times_used_dict[(1, best_pair[0])]
        times_used2 = self.times_used_dict[(2, best_pair[1])]
        return (best_pair[0], component1, times_used1), (best_pair[1], component2, times_used2), best_path
    
    def remove_component_from_set1(self,comp1): 
        component1_pairs = [(comp1, other_comp2) for other_comp2 in self.current_components2]
        for pair_to_remove in component1_pairs: 
            self.aligned_sequence_dict.pop(pair_to_remove)
            self.dtw_distances_dict.pop(pair_to_remove)
        
        self.times_used_dict.pop((1, comp1))
        self.original_components1.pop(comp1)
        self.current_components1.remove(comp1)
        
    def remove_component_from_set2(self,comp2): 
        component2_pairs = [(other_comp1, comp2) for other_comp1 in self.current_components1]
        for pair_to_remove in component2_pairs: 
            self.aligned_sequence_dict.pop(pair_to_remove)
            self.dtw_distances_dict.pop(pair_to_remove)
            
        self.times_used_dict.pop((2, comp2))
        self.original_components2.pop(comp2)
        self.current_components2.remove(comp2)
        
    def add_component_to_set1(self,component1, times_used): 
        comp1_idx = self.current_component_nb1 + 1
        self.current_component_nb1 += 1
        self.current_components1.add(comp1_idx)
        
        self.original_components1[comp1_idx] = component1
        
        self.times_used_dict[(1,comp1_idx)] = times_used
        
        for comp2_idx in self.current_components2: 
            component2 = self.original_components2[comp2_idx] 
            if self.window > 0: 
                aligned_component1, warping_path = dtw.warp(component1, component2, window = self.window, penalty = 0.1)
            else: 
                aligned_component1, warping_path = component1, None
            dist = dtw.distance(component1, component2, window = self.window, penalty = 0.1)
            self.aligned_sequence_dict[(comp1_idx, comp2_idx)] = (aligned_component1, component2, warping_path)
            self.dtw_distances_dict[(comp1_idx, comp2_idx)] = dist
            
   
        
    def add_component_to_set2(self,component2, times_used): 
        comp2_idx = self.current_component_nb2 + 1
        self.current_component_nb2 += 1
        self.current_components2.add(comp2_idx)
        
        self.original_components2[comp2_idx] = component2
        
        self.times_used_dict[(2,comp2_idx)] = times_used
        
        for comp1_idx in self.current_components1: 
            component1 = self.original_components1[comp1_idx]
            if self.window >0: 
                aligned_component1, warping_path = dtw.warp(component1, component2, window = 4, penalty = 0.1)
            else: 
                aligned_component1, warping_path = component1, None
            dist = dtw.distance(component1, component2, window = 0, penalty = 0.1)
            self.aligned_sequence_dict[(comp1_idx, comp2_idx)] = (aligned_component1, component2, warping_path)
            self.dtw_distances_dict[(comp1_idx, comp2_idx)] = dist
    
    def change_times_used_set1(self,comp1_idx, times_used): 
        self.times_used_dict[(1, comp1_idx)] = times_used
        
    def change_times_used_set2(self,comp2_idx, times_used): 
        self.times_used_dict[(2, comp2_idx)] = times_used 