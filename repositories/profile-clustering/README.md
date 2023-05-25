# Scenario generation code
This codebase contains most (if not all) of the code developed by Jonas for the collaboration with EnergyVille.  
It contains code to preprocess, cluster and generate electricity consumption timeseries.
Therefore only parts of this codebase are relevant for scenario generation. 
Note that some parts of this codebase might still be quite messy ;). 

## Subdirectories 
- `archive` contains all of the old code that I did not want to delete yet. So no point in looking at that code! 
- `energyclustering` is the main package directory, all nice-ish code should be in this folder
  - `energyclustering/clustering` all code to do with clustering timeseries. Multiple distance metrics are implemented
  together with convenience methods to easily generate distance matrices or clustering algorithms. 
  - `energyclustering/cobras` COBRAS implementation that is adapted to work in an asynchronous way
  - `energyclustering/data` helper methods to read and parse data 
  - `energyclustering/sampling` the core code for scenario generation 
  - `energyclustering/visualization` some methods for visualization purposes 
  - `energyclustering/webapp` a very simple (badly implemented) webapp to answer queries while running COBRAS in the background. 

So the most relevant part of the code for scenario generation is in `energyclustering/sampling`. 

## Scenario generation explanation

### The models themselves
The models themselves are implemented in day_of_year_samplers.py and samplers.py.  
The code makes use of design patterns such as compositor and decorator to make the code highly flexible (e.g. you can combine different models easily). 

samplers.py contains the simple scenario generation models. 
- The (abstract) BaseSampler class represents a scenario generator that uses some info attributes to generate samples from the given data. Subclasses of this class are the EnergyVilleDaySelectionBaseline (e.g. the procedure of energyville to select a day from the year) and a RandomSamplerBaseline (which just does random sampling).
- The (abstract) ClusteringBaseSampler class represents a scenario generator that uses some info attributes to generate samples from the given data *based on a clustering*. It therefore has an additional method get_cluster_probabilities which assigns a profile to one (or more) of the clusters. Subclasses are the MetadataSampler (clustering on info to sample from data) and the ConsumptionDataSampler (clustering on data and learning a classifier to associate info with the clusters). 

Both these models can be fit using the fit function. To generate scenarios there are two functions:  
- get_sampling_probabilities(info) returns all the non-zero probabilities of every profile in the train data for every of the test metadata. 
- get_samples(info) simply returns a list of samples to take (without any probabilities) for every of the test metadata. 
For most models, get_samples is more efficient as it only takes a predefined number of samples instead of calculating the sampling probability for every possible sample. 

day_of_year_samplers.py contains more complex scenario generation models that take into account both the yearly and daily level.  
These models make use of some of the samplers in the samplers.py file.  

Again, there is an abstract base class that shows the structure of all yeardaysamplers. YearDaySampler contains methods to fit and to generate samples/probabilities. 
The interface is very similar to day_of_year_samplers except that it expects both daily and yearly info and data to be provided. 
The most important subclass is the DailySamplerFromClusterSampler which uses a ClusteringBaseSampler to select which cluster(s) of years to sample from and trains a BaseSampler per cluster to select the days from that cluster. 

## Other submodules
There is some inspection code in energyclustering/sampling/inspection. But this could be very old and not working anymore! 
There is some evaluation code in energyclustering/sampling/evaluation. One file contains the implementation of the energyscore while the other contains the evaluation framework (using dask). Note that since one of the last updates this is not very stable! For some reason dask seems to misbehave, best to install a dask version that starts with 2021 (e.g. conda install dask=2021)

## Small example 
To see how the models are build you can take a look at notebook clean_notebooks/2022_10_26_paper_experiments_after_refactor/01.overall_comparison.ipynb.  

However, again note that not all these notebooks will work! 