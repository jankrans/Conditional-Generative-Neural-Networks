class RandomSampler:
    """
        UNUSED?
        Just a random sampler that samples each sample with equal probability
    """

    def __init__(self):
        self.n_samples = None
        self.consumption_data_index = None

    def fit(self, info, consumption_data, clustering=None, cluster_centroids=None):
        self.n_samples = consumption_data.shape[0]
        self.consumption_data_index = consumption_data.index

    def get_sampling_probabilities(self, test_info):
        return pd.DataFrame(np.full((test_info.shape[0], self.n_samples),1 / self.n_samples), index = test_info.index, columns = self.consumption_data_index)

    def get_sampling_probabilities_daily(self, test_info):
        return list(item for _, item in self.get_sampling_probabilities(test_info).iterrows())


class ConsumptionDataSamplerWithValidation:
    """
        A class that represent an object that can sample from a clustering based on consumption data

        A probabilistic (or deterministic) classifier is used to learn to assign an 'instance' to the correct cluster based on exogenous attributes.

        This class can be used for yearly and for daily data (e.g. any other type of data as well)
     """

    def __init__(self, classifier, clusterer, validation_perc = 0.25, info_preprocessing = None, fillna= True, random_state = None):
        self.classifier = classifier
        self.clusterer = clusterer

        self.info_preprocessing = info_preprocessing

        self.clustering = None
        self.consumption_data = None

        self.validation_perc = 0.25
        self.random_state = random_state


        self.fillna = fillna

    def fit(self, info, consumption_data):
        # preprocess info if necessary
        if self.info_preprocessing is not None:
            info = self.info_preprocessing.fit_transform(info)

        # save the consumption data for sampling
        if self.fillna:
            self.consumption_data = consumption_data.fillna(0)

        # fit the clustering on all consumption data
        self.clusterer.old_fit(self.consumption_data)
        self.clustering = pd.Series(self.clusterer.labels_, index = self.consumption_data.index)

        # split the consumption data into train and validation set
        y_train, y_valid = train_test_split(self.clustering, test_size = self.validation_perc, random_state= self.random_state)

        train_info, valid_info = info.loc[y_train.index] , info.loc[y_valid.index]
        # fit the classifier on the training set only
        self.classifier.old_fit(train_info, y_train)

        # effective alpha's for cost complexity pruning
        effective_alphas = self.classifier.cost_complexity_pruning_path(train_info, y_train).ccp_alphas

        # train decision trees deeper and deeper until validation set accuracy decreases
        best_classifier = None
        validation_set_score = None
        for alpha in effective_alphas[::-1]:
            self.classifier.set_params(ccp_alpha = alpha)
            self.classifier.old_fit(train_info, y_train)
            y_pred_probs = self.classifier.predict_proba(valid_info)
            all_clusters = np.sort(self.clustering.unique())
            correct_y_pred_probs = np.zeros((y_pred_probs.shape[0], all_clusters.shape[0]))
            correct_y_pred_probs[:, self.classifier.classes_] = y_pred_probs
            score = log_loss(y_valid, correct_y_pred_probs, labels = all_clusters)
            if validation_set_score is None or validation_set_score <= score:
                validation_set_score = score
                best_classifier = deepcopy(self.classifier)
            else:
                break
        self.classifier = best_classifier
        best_classifier.fit(info, self.clusterer.labels_)





    def get_cluster_probabilities(self, test_info):
        return self.classifier.predict_proba(test_info)

    def get_sampling_probabilities(self, test_info):
        # preprocess info if necessary
        if self.info_preprocessing is not None:
            test_info = self.info_preprocessing.transform(test_info)

        # let the classifier predict the probabilities of being part of the cluster

        cluster_probabilities = self.classifier.predict_proba(test_info)

        # calculate the sample probabilities when sampling uniformly from the chosen cluster
        sample_probabilities = cluster_probabilities_to_sample_probabilities(cluster_probabilities, self.clustering)
        return pd.DataFrame(sample_probabilities, index = test_info.index, columns = self.consumption_data.index)

    def get_sampling_probabilities_daily(self, test_info):
        return list(item for _, item in self.get_sampling_probabilities(test_info).iterrows())
