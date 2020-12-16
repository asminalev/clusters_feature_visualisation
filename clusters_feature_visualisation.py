from scipy.stats import norm
def plot_clustering_feature_density(X,predicted_labels, fig_size=(10,8)):

        X = pd.DataFrame(X)

        N_clusters = len(set(predicted_labels))

        # prepare subplot_axis_indices:
        c1 = N_clusters % 3
        c2 = N_clusters % 2
        
        N_col = 3 if c1<=c2 else 2

        N_row = int(N_clusters / N_col) + (N_clusters % N_col)
        r,c = [np.arange(n) for n in [N_row,N_col]]


        ix = list(it.product(r,c))

        fig, axes = plt.subplots(N_row, N_col, figsize=fig_size)

        if N_row == 1:
            axes = axes.reshape(1,-1)
        
        # iterate over the labels
        for nplt, lbl in enumerate(set(predicted_labels)):

            # exclude weighted_score
            cluster_df = X.loc[predicted_labels == lbl]
            
            # Draw the density plot
            for i, data in enumerate(cluster_df.iteritems()):

                m_name, metric = data
                sns.distplot(metric, hist = False, kde = True,norm_hist=True,
                             kde_kws = {'linewidth': 2},
                             label = m_name,ax=axes[ix[lbl]])


                # Plot formatting
                axes[ix[lbl]].legend(prop={'size': 8}, loc='upper right',title = 'n_customers: ' + str(cluster_df.shape[0])) # bbox_to_anchor=(0.25,0.25), title = 'Clusters',
                axes[ix[lbl]].title.set_text('Cluster: ' + str(lbl) + ' | Density_Plot')
                axes[ix[lbl]].set_xlabel('Log Scaled Feature Values')
                
                if nplt == 0:
                    axes[ix[lbl]].set_ylabel('Probability_Density')
        plt.show()        
        print('\n'*2, '-'*100, '\n'*2)
    


# ============
# Set up cluster parameters
# ============

plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

''' 

your data must be only including numeric features.

'''

XDS = your_data.copy() 

datasets = [
    (XDS.values, {'damping': .77, 'preference': -240,
                     'quantile': .18, 'n_clusters': 3,
                     'min_samples': 20, 'xi': 0.25})]
    
    
YDATA = []
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)
    
    print(i_dataset, params)
    X = dataset

    # normalize dataset for easier parameter selection
    # = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = sk_cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    
    two_means = sk_cluster.MiniBatchKMeans(n_clusters=params['n_clusters']) 
    
    
    spectral = sk_cluster.SpectralClustering(
    n_clusters=params['n_clusters'], eigen_solver='arpack',
    affinity="nearest_neighbors")
    

    gmm = mixture.GaussianMixture(
    n_components=params['n_clusters'], covariance_type='full')
    
    
    ward = sk_cluster.AgglomerativeClustering(
    n_clusters=params['n_clusters'], linkage='ward',
    connectivity=connectivity)
    
    average_linkage = sk_cluster.AgglomerativeClustering(
    linkage="average", affinity="euclidean",
    n_clusters=params['n_clusters'], connectivity=connectivity)
    

    ms = sk_cluster.MeanShift(bandwidth=bandwidth, bin_seeding=False)
    
          
    ward = sk_cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)

    clustering_algorithms = (
        
        ('MiniBatchKMeans', two_means),
        ('SpectralClustering', spectral),
        ('GaussianMixture', gmm),
        ('AgglomerativeClustering', average_linkage),
        ('MeanShift', ms),
        ('Ward', ward)
    )
    
    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        print(set(y_pred))
        print('i_dataset: ', i_dataset, ' algorithm: ', name)
        
        YDATA.append([name,y_pred])
        
        plot_clustering_feature_density(XDS, y_pred, fig_size=(15,8))
        plt.show()
        