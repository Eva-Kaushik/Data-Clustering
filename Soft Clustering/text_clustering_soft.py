import functions_categorization_new as category

def getClusterDetails(data):
    
    # Text Preprocessing
    data['Ticket Title Preprocessed'] = category.text_preprocessing(data['Ticket Title'],include_verbs = False)

    # Clustering
    df_un_tfidf=category.word_matrix(column_name=data['Ticket Title Preprocessed'],min_df_uni_thr=5,min_df_bi_thr=10)
    dense_tfidf,K_NGrams=category.optimum_clusters(df_un_tfidf)
    dm_tfidf_dfm_rep_docs,K_N_Comp=category.LSA_optimum_components(dense_tfidf,K_NGrams,Exp_varaince_LSA_thr=0.8)

    # cluster analysis
    clustered_data = category.cluster_file_soft(data,dm_tfidf_dfm_rep_docs,K_N_Comp,K_NGrams,3)

    # Get the cluster names
    clustered_data_with_names = category.naming_clusters_soft(clustered_data, actual_column_name='Ticket Title')

    # Merging Data
    clustered_data['ClusterName1'] = clustered_data_with_names['Cluster']
    clustered_data['ClusterName2'] = clustered_data_with_names['Cluster2']
    clustered_data['ClusterName1_proba'] = clustered_data_with_names['Proba1']
    clustered_data['ClusterName2_proba'] = clustered_data_with_names['Proba2']
	
    return clustered_data