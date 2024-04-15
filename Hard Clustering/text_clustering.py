import functions_categorization_new as category

def getClusterDetails(data):
    
    # Text Preprocessing
    data['Ticket Title Preprocessed'] = category.text_preprocessing(data['Ticket Title'],include_verbs = False)

    # Clustering
    df_un_tfidf=category.word_matrix(column_name=data['Ticket Title Preprocessed'],min_df_uni_thr=5,min_df_bi_thr=10)
    dense_tfidf,K_NGrams=category.optimum_clusters(df_un_tfidf)
    dm_tfidf_dfm_rep_docs,K_N_Comp=category.LSA_optimum_components(dense_tfidf,K_NGrams,Exp_varaince_LSA_thr=0.8)

    # cluster analysis
    clustered_data,lsa_doc_concepts_df = category.cluster_file(data,dm_tfidf_dfm_rep_docs,K_N_Comp,K_NGrams,3)

    # Get the cluster names
    clustered_data_with_names = category.naming_clusters(clustered_data,lsa_doc_concepts_df, actual_column_name='Ticket Title')

    # Merging Data
    clustered_data[['ClusterName', 'Cluster Medoid_1', 'Cluster Medoid_2', 'Cluster Medoid_3']] = clustered_data_with_names[['Cluster', 'Cluster Medoid_1', 'Cluster Medoid_2', 'Cluster Medoid_3']]
	
    return clustered_data