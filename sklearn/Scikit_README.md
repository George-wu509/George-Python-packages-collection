# George-Python-packages-collection



4. Scikit-learn (sklearn) - Machine learning in Python
--------------------------------
- 4.1 sklearn.`base`  - **base classes and utility functions**
  - sklearn.base.`BaseEstimator` - Base class for all estimators in scikit-learn
  - sklearn.base.`ClassifierMixin` - Mixin class for all classifiers in scikit-learn
  - sklearn.base.`ClusterMixin` - Mixin class for all cluster estimators in scikit-learn.
  - sklearn.base.`RegressorMixin` - 	Mixin class for all regression estimators in scikit-learn
  - sklearn.base.`TransformerMixin` - Mixin class for all transformers in scikit-learn
  - [Fn] sklearn.base.`clone` - Constructs a new estimator with the same parameters
  
- 4.2 sklearn.`cluster` - **Clustering**
  - sklearn.cluster.`AffinityPropagation` - Perform Affinity Propagation Clustering of data
  - sklearn.cluster.`AgglomerativeClustering` - Agglomerative Clustering
  - sklearn.cluster.`Birch` - Implements the Birch clustering algorithm.
  - sklearn.cluster.`DBSCAN` - Perform DBSCAN clustering from vector array or distance matrix
  - sklearn.cluster.`FeatureAgglomeration` - Agglomerate features
  - sklearn.cluster.`KMeans` - K-Means clustering
  - sklearn.cluster.`MiniBatchKMeans` - Mini-Batch K-Means clustering
  - sklearn.cluster.`MeanShift` - Mean shift clustering using a flat kernel.
  - sklearn.cluster.`SpectralClustering` - 	Apply clustering to a projection to the normalized laplacian
  - [Fn] sklearn.cluster.`estimate_bandwidth` - Estimate the bandwidth to use with the mean-shift algorithm.
  - [Fn] sklearn.cluster.`k_means` - K-means clustering algorithm
  - [Fn] sklearn.cluster.`ward_tree` - Ward clustering based on a Feature matrix
  - [Fn] sklearn.cluster.`affinity_propagation` - Perform Affinity Propagation Clustering of data
  - [Fn] sklearn.cluster.`dbscan` - Perform DBSCAN clustering from vector array or distance matrix
  - [Fn] sklearn.cluster.`mean_shift` - Perform mean shift clustering of data using a flat kernel
  - [Fn] sklearn.cluster.`spectral_clustering` - Apply clustering to a projection to the normalized laplacian
  
- 4.3 sklearn.`cluster.bicluster` - **Biclustering**
  - sklearn.cluster.bicluster.`SpectralBiclustering` - Spectral biclustering
  - sklearn.cluster.bicluster.`SpectralCoclustering` - Spectral Co-Clustering algorithm

- 4.4 sklearn.`covariance` - **Covariance Estimators**  
  - sklearn.covariance.`EmpiricalCovariance` - Maximum likelihood covariance estimator
  - sklearn.covariance.`EllipticEnvelope` - An object for detecting outliers in a Gaussian distributed dataset.
  - sklearn.covariance.`GraphLasso` - Sparse inverse covariance estimation with an l1-penalized estimator
  - sklearn.covariance.`GraphLassoCV` - Sparse inverse covariance w/ cross-validated choice of the l1 penalty
  - sklearn.covariance.`LedoitWolf` - LedoitWolf Estimator
  - sklearn.covariance.`MinCovDet` - Minimum Covariance Determinant (MCD): robust estimator of covariance
  - sklearn.covariance.`OAS` - Oracle Approximating Shrinkage Estimator
  - sklearn.covariance.`ShrunkCovariance` - Covariance estimator with shrinkage
  - sklearn.covariance.`empirical_covariance` - Computes the Maximum likelihood covariance estimator
  - sklearn.covariance.`ledoit_wolf(` - Estimates the shrunk Ledoit-Wolf covariance matrix
  - sklearn.covariance.`shrunk_covariance` - Calculates a covariance matrix shrunk on the diagonal
  - sklearn.covariance.`oas` - Estimate covariance with the Oracle Approximating Shrinkage algorithm.
  - sklearn.covariance.`graph_lasso` -l1-penalized covariance estimator
  - [Fn] sklearn.covariance.`EmpiricalCovariance` - S

- 4.5 sklearn.`model_selection` - **Model Selection**
  - sklearn.model_selection.`KFold` - K-Folds cross-validator
  - sklearn.model_selection.`GroupKFold` - K-fold iterator variant with non-overlapping groups
  - sklearn.model_selection.`StratifiedKFold` - Stratified K-Folds cross-validator
  - sklearn.model_selection.`LeaveOneGroupOut` - Leave One Group Out cross-validator
  - sklearn.model_selection.`LeavePGroupsOut` - Leave P Group(s) Out cross-validator
  - sklearn.model_selection.`LeaveOneOut` - Leave-One-Out cross-validator
  - sklearn.model_selection.`LeavePOut` - Leave-P-Out cross-validator
  - sklearn.model_selection.`ShuffleSplit` - Random permutation cross-validator
  - sklearn.model_selection.`GroupShuffleSplit` - Shuffle-Group(s)-Out cross-validation iterator
  - sklearn.model_selection.`StratifiedShuffleSplit` - Stratified ShuffleSplit cross-validator
  - sklearn.model_selection.`PredefinedSplit` - Predefined split cross-validator
  - sklearn.model_selection.`TimeSeriesSplit`- Time Series cross-validator    
    - **Splitter Functions** 
  - sklearn.model_selection.`train_test_split` - Split arrays or matrices into random train and test subsets
    - **Hyper­parameter optimizers**  
  - sklearn.model_selection.`GridSearchCV` - Exhaustive search over specified parameter values for an estimator.
  - sklearn.model_selection.`RandomizedSearchCV` - Randomized search on hyper parameters.
  - sklearn.model_selection.`ParameterGrid` - Grid of parameters with a discrete number of values for each.
  - sklearn.model_selection.`ParameterSampler` - Generator on parameters sampled from given distributions.  
    - **Model validation**   
  - sklearn.model_selection.`cross_val_score` - Evaluate a score by cross-validation
  - sklearn.model_selection.`cross_val_predict` - Generate cross-validated estimates for each input data point
  - sklearn.model_selection.`permutation_test_score` - Evaluate the significance of a cross-validated score with permutations
  - klearn.model_selection.`learning_curve` - Learning curve.
  - sklearn.model_selection.`validation_curve` - Validation curve.

  
- 4.6 sklearn.`datasets` - **Datasets**  
- 4.7 sklearn.`decomposition` - **Matrix Decomposition**  
- 4.8 sklearn.`dummy` - **Dummy estimators**
- 4.9 sklearn.`ensemble` - **Ensemble Methods**  
- 4.10 sklearn.`exceptions` - **Exceptions and warnings**  
- 4.11 sklearn.`feature_extraction` - **Feature Extraction**  
- 4.12 sklearn.`feature_selection` - **Feature Selection**  
- 4.13 sklearn.`gaussian_process` - **Gaussian Processes**  
- 4.14 sklearn.`isotonic` - **Isotonic regression**  
- 4.15 sklearn.`kernel_approximation` - **Kernel Approximation**  
- 4.16 sklearn.`kernel_ridge` - **Kernel Ridge Regression**  
- 4.17 sklearn.`discriminant_analysis` - **Discriminant Analysis**  
- 4.18 sklearn.`linear_model` - **Generalized Linear Models**  
- 4.19 sklearn.`manifold` - **Manifold learning**  
- 4.20 sklearn.`metrics` - **Metrics**  
- 4.21 sklearn.`mixture` - **Gaussian Mixture Models**  
- 4.22 sklearn.`multiclass` - **Multiclass and multilabel classification**  
- 4.23 sklearn.`multioutput` - **Multioutput regression and classification**  
- 4.24 sklearn.`naive_bayes` - **Naive Bayes**  
- 4.25 sklearn.`neighbors` - **Nearest Neighbors**  
- 4.26 sklearn.`neural_network` - **Neural network models**  
- 4.27 sklearn.`calibration` - **Probability Calibration**  
- 4.28 sklearn.`cross_decomposition` - **Cross decomposition**  
- 4.29 sklearn.`pipeline` - **Pipeline**  
- 4.30 sklearn.`preprocessing` - **Preprocessing and Normalization**  
- 4.31 sklearn.`random_projection` - **Random projection**  
- 4.32 sklearn.`semi_supervised` - **Semi-Supervised Learning**  
- 4.33 sklearn.`scm` - **Support Vector Machines**  
- 4.34 sklearn.`tree` - **Decision Trees**  
- 4.35 sklearn.`utils` - **Utilities**  

