# coding: utf-8

#import Packages required for the Analysis
import re
import pandas as pd 
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold  
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from scipy.sparse import  hstack
from sklearn.metrics import  pairwise_distances
import numpy as np
import nltk
from nltk.util import ngrams
from nltk.corpus import RegexpTokenizer as regextoken
import spacy
nlp = spacy.load('en_core_web_sm')

Bigram=False
    
def text_preprocessing(document, include_verbs = True, include_numbers = True):
    """
    Description: Performs basic text preprocessing steps 
    Input: 
    1. document - raw free flow text column from data
    2. include_verbs - True(default): Allowed word types include 'ADJ', 'NOUN', 'PROPN, 'ADV', 'VERB'
                     - False: Allowed word types include 'ADJ','NOUN','PROPN'
    3. include_numbers - True(default): Include tokens that are actually numbers but are tagged as 'NOUN'
                       - False: Remove all tokens that have only numbers and special characters
    Output: 
    1. processed_text - stopword free and lemmatized version of text
    """
    # Convert to string
    document  = document.astype(str)
    # Allowed word types
    if(include_verbs):
        allowed_word_types = ['ADJ','NOUN','PROPN','ADV','VERB']
    else:
        allowed_word_types = ['ADJ','NOUN','PROPN']
    # Stop words in danish that we should remove 
    #stop_words = list(spacy.lang.da.stop_words.STOP_WORDS)
    stop_words = list(spacy.lang.en.stop_words.STOP_WORDS)
    # Domain specific stop words you might want to remove
    domainSpecificStopwords = []
    stop_words.extend(domainSpecificStopwords)
    processed_text = []
    for t in document:
        t = str(t)
        #print(t)
        doc = nlp(t)
        #print(doc)
        all_words = []
        # Reduced token list: Allowed only specific word types, length has to be more than 1 and must not be a stop word
        words = [token.lemma_.strip().lower() for token in doc if token.pos_ in allowed_word_types and len(token.text)>1 and token.text not in stop_words]
        # Remove tokens that contain only numbers or special characters
        if(include_numbers):
            all_words = words
        else:
            for wor in words:
                if(re.search(r"^[\d/.:-]+$",wor)):
                    pass
                else:
                    all_words.append(wor)
        summary = ' '.join(all_words)
        #print(summary)
        #added for non english
        #summary = Non_English_Weightage_NLTK(summary)
        #print(summary)
        processed_text.append(summary)
    return processed_text

# Function for English and NON split
def Non_English_Weightage_NLTK(document):
    # English Library
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    all_words=[]
    summary=""
    tokenizer = regextoken(r'\w+')
    words = tokenizer.tokenize(document)
    for word in words:
        #if word in english_vocab or word in Base_word :
        if word in english_vocab :
            all_words.append(word)
        else: 
            all_words.append(word) 
            all_words.append(word) 
    summary=' '.join(all_words)      
    return summary

#Building Pipeline for word matrix
def word_matrix(column_name,min_df_uni_thr,min_df_bi_thr):
    """
    Description: tf-idf matrix of text column 
    Input: 
    1. column_name - column to be converted into tf-idf matrix
    2. min_df_uni_thr - unigram threshold
    3. min_df_bi_thr - bigram threshold
    
    Output: 
    1. dm_tfidf - tf-idf matrix
    """
    # Domain specific stop words to be added to spacy stop words list
    domainSpecificStopwords = []
    # Stop words in danish that we should remove 
    #my_stop_words = list(spacy.lang.da.stop_words.STOP_WORDS)
    my_stop_words = list(spacy.lang.en.stop_words.STOP_WORDS)
    my_stop_words.extend(domainSpecificStopwords)
    count_vect = CountVectorizer(min_df=min_df_uni_thr,stop_words=my_stop_words) #max features
    X_train_counts_UNI = count_vect.fit_transform(column_name.astype("str"))
    tfidf_transformer = TfidfTransformer()
    dm_tfidf_unigram = tfidf_transformer.fit_transform(X_train_counts_UNI)

    if Bigram==True:
        text_clf_bigram = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2),min_df_bi=min_df_bi_thr)),('tfidf', TfidfTransformer()),])
        text_clf_bigram.fit(column_name.astype("str"))
        dm_tfidf_bigram=text_clf_bigram.transform(column_name.astype("str"))
        dm_tfidf=hstack([dm_tfidf_unigram,dm_tfidf_bigram])

    else:
        dm_tfidf=dm_tfidf_unigram
    
    return(dm_tfidf)

## Find Initial number of clusters based on TFIDF matrix
def optimum_clusters(dm_tfidf):
    """
    Description: Initial number of clusters using thumb rule(sparsity of matrix) 
    Input: 
    1. dm_tfidf - tf-idf matrix
    
    Output: 
    1. dm_tfidf_dfm - dense matrix
    2. K_NGrams - number of clusters using thumb rule
    """
    dm_tfidf_dfm=pd.DataFrame(dm_tfidf.todense())
    dnm=dm_tfidf_dfm.astype(bool).sum(axis=0).sum()
    nmr=dm_tfidf_dfm.shape[0]*dm_tfidf_dfm.shape[1]
    K_NGrams=round(nmr/dnm,0)
    return dm_tfidf_dfm, K_NGrams

def LSA_optimum_components(dm_tfidf_dfm,K_NGrams,Exp_varaince_LSA_thr):
    """
    Description: Get optimum LSA components that explains required variance 
    Input: 
    1. dm_tfidf_dfm - dense matrix
    
    Output: 
    1. dm_tfidf_dfm_rep_docs -  matrix after removing sparse documents
    2. K_N_Comp - number of LSA components
    """
    dm_tfidf_dfm_rep_docs=dm_tfidf_dfm.loc[~dm_tfidf_dfm.apply(lambda row: (row==0).all(), axis=1)]
    scores = []
    N_Com_val=[]
    Start=int(K_NGrams)
    end=dm_tfidf_dfm_rep_docs.shape[1]
    for k in range(Start,end,50):
        svd = TruncatedSVD(n_components=k)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X_lsa = lsa.fit_transform(dm_tfidf_dfm_rep_docs)
        explained_variance = svd.explained_variance_ratio_.sum()
        scores.append(int(explained_variance * 100))
        N_Com_val.append(k)
        if explained_variance > Exp_varaince_LSA_thr:
            break    # break here

    Score_Df = pd.DataFrame({'N-com_val': N_Com_val,'Expl_var': scores})
    if Score_Df.shape[0] == 0:
        K_N_Comp = K_NGrams
    else:
        K_N_Comp=Score_Df.loc[Score_Df.shape[0]-1,'N-com_val']
    return dm_tfidf_dfm_rep_docs,K_N_Comp

def cluster_file(data,dm_tfidf_dfm_rep_docs,K_N_Comp,K_NGrams,n_init_K):
    """
    Description: Clustering of documents 
    Input: 
    1. data - initial data
    2. dm_tfidf_dfm_rep_docs - matrix after removing sparse documents
    3. K_N_Comp - number of LSA components
    4. K_NGrams - number of clusters using thumb rule
    5. n_init_K - Number of times the k-means algorithm will run with different
    centroid seeds.
    
    Output: 
    clustered data frame

    """
    if K_N_Comp!= K_NGrams:
        svd = TruncatedSVD(n_components=int(K_N_Comp), random_state=42)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        lsa_doc_concepts = lsa.fit_transform(dm_tfidf_dfm_rep_docs)
    else:
        lsa_doc_concepts = dm_tfidf_dfm_rep_docs
    
    km = KMeans(n_clusters=int(K_NGrams), init='k-means++', n_init=n_init_K,algorithm='full', random_state=123)
    km.fit(lsa_doc_concepts)
    predict=km.predict(lsa_doc_concepts)
    dm_tfidf_dfm_rep_docs['ClusterId']=pd.Series(predict, index=dm_tfidf_dfm_rep_docs.index)
    clusterid=dm_tfidf_dfm_rep_docs['ClusterId'].to_frame()
    
    lsa_doc_concepts_df = pd.DataFrame(lsa_doc_concepts, index=dm_tfidf_dfm_rep_docs.index)
    lsa_doc_concepts_df['ClusterId'] = predict
    
    data_clusterid=data.join(clusterid)
#    data_clusterid.to_csv('unassigned_clusters.csv', sep=',',index=False)
#    data_clusterid.count(axis=0)
    return data_clusterid, lsa_doc_concepts_df

def medoid_name(df, lsa_doc_concepts_df, clust, actual_column_name):
    distmatrix = pairwise_distances(lsa_doc_concepts_df.loc[lsa_doc_concepts_df['ClusterId'] == clust].drop('ClusterId', axis=1))
    #median_index = np.argmin(distmatrix.sum(axis=0))
    A = np.array(distmatrix.sum(axis = 0))
    median_index_list = A.argsort()[:3]
    # Get the corresponding text data for the index
    cluster_names = [df.loc[k,actual_column_name] for k in median_index_list]
    if len(cluster_names) == 3:
        pass
    elif len(cluster_names) == 2:
        cluster_names.append('')
    elif len(cluster_names) == 1:
        cluster_names.extend(['',''])
    else:
        cluster_names = ['','','']
    return cluster_names
      
    
# Naming the clusters when the unnamed cluster data is passed as an input
def naming_clusters(df,lsa_doc_concepts_df,actual_column_name):
    '''
    Args:
        df: unnamed clustered data
        lsa_doc_concepts_df: Input data along with the clusterId 
        actual_column_name: Actual column name
    returns:
        clustered_data: Actual column along with the cluster name
    '''
    clustered_data = df[[actual_column_name,'ClusterId']]
    # As we removed documents which are not contributing to any features, these are not assigned to any cluster
    clustered_data['ClusterId'].fillna('unable to assign cluster label',inplace = True)
    # Remove special character from actual column name   
    clustered_data.loc[:, actual_column_name] = clustered_data[actual_column_name].apply(removeSpecialCharacters)
    #clustered_data[actual_column_name] = clustered_data[actual_column_name].apply(lambda x: removeSpecialCharacters(x) if type(x)==str else x)
    # cluster numbers
    cluster_id = clustered_data['ClusterId'].unique()
    
    clustered_data.loc[:,'Cluster Medoid_1'] = clustered_data['ClusterId']
    clustered_data.loc[:,'Cluster Medoid_2'] = clustered_data['ClusterId']
    clustered_data.loc[:,'Cluster Medoid_3'] = clustered_data['ClusterId']
    
    # Finding name to each cluster 
    cluster_name_given = {}
    cluster_name_medoid_1 = {}
    cluster_name_medoid_2 = {}
    cluster_name_medoid_3 = {}
    for clust in cluster_id:
        if clust != 'unable to assign cluster label':
            _cluster = []
            
            # slicing data for respective clusters
            new_df = clustered_data[clustered_data['ClusterId'] == clust].reset_index(drop = True)
            
            # Get the cluster mediods for each cluster
            medoids_list = medoid_name(new_df,lsa_doc_concepts_df, clust, actual_column_name)
            cluster_name_medoid_1[int(clust)] = medoids_list[0]
            cluster_name_medoid_2[int(clust)] = medoids_list[1]
            cluster_name_medoid_3[int(clust)] = medoids_list[2]
            
            for doc in new_df[actual_column_name].fillna(''):
                _cluster.append(doc)
                    
            # Finding name for that cluster
            name = clustername(_cluster) #topngram_clustername(_cluster)
            if len(name) != 0:
                cluster_name_given[int(clust)] = name[0]

            else:
                cluster_name_given[int(clust)] = str(clust)+"_test_name" 
        
    # Renaming Columns
    clustered_data.rename(columns = {'ClusterId':'Cluster'}, inplace = True)
    
    # Assigning names for each cluster 
    clustered_data.loc[:, 'Cluster'] = clustered_data['Cluster'].map(cluster_name_given).fillna('unable to assign cluster label')
    clustered_data.loc[:,'Cluster Medoid_1'] =  clustered_data['Cluster Medoid_1'].map(cluster_name_medoid_1).fillna('unable to assign cluster label')
    clustered_data.loc[:,'Cluster Medoid_2'] =  clustered_data['Cluster Medoid_2'].map(cluster_name_medoid_2).fillna('unable to assign cluster label')
    clustered_data.loc[:,'Cluster Medoid_3'] =  clustered_data['Cluster Medoid_3'].map(cluster_name_medoid_3).fillna('unable to assign cluster label')
    
    return clustered_data    

def removeSpecialCharacters(text):
    text = str(text)
    text = text.lower()
    text = re.sub('[^A-Za-z0-9 ]+', '', str(text))
    
    return text
    
# Naming a cluster where all the data points assigned to the particular cluster is passed as a parameter
def clustername(text):
    '''
    text : text data as pandas series on which name is calculated. 
    '''
    cluster_name = []
    # finding unigrams, bigrams, trigrams
    unigram1 = n_grams_dict(text, n = 1, thresh = 0.75)#0.10
    bigram1 = n_grams_dict(text, n = 2, thresh = 0.65) #0.40
    trigram1 = n_grams_dict(text, n = 3, thresh = 0.6)#0.30
    fourgram1 = n_grams_dict(text, n = 4, thresh = 0.55)#0.30
    fivegram1 = n_grams_dict(text, n = 5, thresh = 0.5)#0.30
        
    # Get the words
    if len(unigram1)!=0:
        unigram_word = next(iter(unigram1))
    else: 
        unigram_word = ''
    if len(bigram1)!= 0:
        bigram_word = next(iter(bigram1))
    else:
        bigram_word = ''
    if len(trigram1)!=0:
        trigram_word = next(iter(trigram1))
    else:
        trigram_word = ''
    if len(fourgram1)!= 0:
        fourgram_word = next(iter(fourgram1))
    else:
        fourgram_word = ''
    if len(fivegram1)!=0:
        fivegram_word = next(iter(fivegram1))
    else:
        fivegram_word = ''

    # Naming the clusters using the below process
    if fivegram_word!= '' and fourgram_word!= '' and trigram_word!= '' and bigram_word!= '' and unigram_word!= '' and fivegram1[fivegram_word] >= fourgram1[fourgram_word]:
        cluster_name.append(fivegram_word)
    elif fivegram_word== '' and fourgram_word!= '' and trigram_word!= '' and bigram_word!= '' and unigram_word!= '' and fourgram1[fourgram_word] >= trigram1[trigram_word]:
        cluster_name.append(fourgram_word)
    elif fivegram_word== '' and fourgram_word== '' and trigram_word!= '' and bigram_word!= '' and unigram_word!= '' and trigram1[trigram_word] >= bigram1[bigram_word]:
        cluster_name.append(trigram_word)
    elif fivegram_word== '' and fourgram_word== '' and trigram_word == '' and bigram_word!= '' and unigram_word!= '' and bigram1[bigram_word] >= unigram1[unigram_word]:
        cluster_name.append(bigram_word)
    #-------------------------------------------------
    elif fivegram_word!= '' and fourgram_word!= '' and trigram_word!= '' and bigram_word!= '' and unigram_word!= '' \
        and unigram_word not in trigram_word and bigram_word not in trigram_word and unigram_word not in bigram_word:
        name = unigram_word + '/'+bigram_word +'/'+trigram_word
        cluster_name.append(name)
    elif fivegram_word!= '' and fourgram_word!= '' and trigram_word!= '' and bigram_word!= '' and unigram_word!= '' \
        and unigram_word not in trigram_word and bigram_word not in trigram_word:
        name = unigram_word +'/'+trigram_word
        cluster_name.append(name)
    #-------------------------------------------------
    elif fivegram_word!= '' and fourgram_word!= '' and trigram_word!= '' and bigram_word!= '' and unigram_word!= ''  \
        and unigram_word in fivegram_word and bigram_word in fivegram_word and trigram_word in fivegram_word and fourgram_word in fivegram_word:
        cluster_name.append(fivegram_word)
    elif fivegram_word == '' and fourgram_word!= '' and trigram_word!= '' and bigram_word!= '' and unigram_word!= ''  \
        and unigram_word in fourgram_word and bigram_word in fourgram_word and trigram_word in fourgram_word :
        cluster_name.append(fourgram_word)
    elif fivegram_word == '' and fourgram_word == '' and trigram_word!= '' and bigram_word!= '' and unigram_word!= ''  \
        and unigram_word in trigram_word and bigram_word in trigram_word:
        cluster_name.append(trigram_word)
    elif fivegram_word == '' and fourgram_word == '' and trigram_word == '' and bigram_word!= '' and unigram_word!= ''  \
        and unigram_word in bigram_word:
        cluster_name.append(bigram_word)
    #-------------------------------------------------
    elif fivegram_word!= '':
        cluster_name.append(fivegram_word)
    elif fourgram_word!= '':
        cluster_name.append(fourgram_word)
    elif trigram_word!= '':
        cluster_name.append(trigram_word)
    elif bigram_word!= '':
        cluster_name.append(bigram_word)
    elif unigram_word!= '':
        cluster_name.append(unigram_word)
    else:
        cluster_name.append('no_name')

    return cluster_name
    
def n_grams_dict(texts, n, thresh):
    '''
    texts : text data as a pandas series on which ngrams are calculated.
    n : int, number of grams to be considered
    thresh : float, minumum percentage of total texts to be considered.
    spacy.load(): is a convenience wrapper that reads the language ID and pipeline components from a model's meta.json, 
    initialises the Language class, loads in the model data and returns it.
    '''
    # loading german stop words
    #stop_words = list(spacy.lang.da.stop_words.STOP_WORDS)
    stop_words = list(spacy.lang.en.stop_words.STOP_WORDS)
    updated_text = []
    # unigram
    if n == 1:
        # Considering only nouns as unigrams
        for i in texts:
            doc = nlp(i)
            text_noun = []
            for token in doc:
                # checking pos tag 
                if token.pos_ == 'NOUN':
                    text_noun.append(token.text)
            # appending all unigrams for each sent
            updated_text.append(" ".join(str(e) for e in text_noun))
                
        # finding unigrams from noun words
        #total_ngrams = list(ngrams([word for doc in updated_text for word in doc.split()],n = n))
        total_ngrams = [list(ngrams(t.split(),n = n)) for t in updated_text]
        merged = list(itertools.chain(*total_ngrams))
        total_ngrams = merged

        threshold = round(thresh * len(texts))
        # calculating frequency counts of unigrams
        total_ngrams_dict = dict(nltk.FreqDist(total_ngrams).most_common())
        
    if n == 2:
        # finding bigrams from data
        #total_ngrams = list(ngrams([word for doc in texts for word in doc.split()],n = n))
        total_ngrams = [list(ngrams(t.split(),n = n)) for t in texts]
        merged = list(itertools.chain(*total_ngrams))
        total_ngrams = merged
        threshold = round(thresh * len(texts))
        # calculating frequency counts of unigrams
        total_ngrams_dict = dict(nltk.FreqDist(total_ngrams).most_common())
    
    if n == 3:
        # finding bigrams from data
        #total_ngrams = list(ngrams([word for doc in texts for word in doc.split()],n = n))
        total_ngrams = [list(ngrams(t.split(),n = n)) for t in texts]
        merged = list(itertools.chain(*total_ngrams))
        total_ngrams = merged
        threshold = round(thresh * len(texts))
        # calculating frequency counts of unigrams
        total_ngrams_dict = dict(nltk.FreqDist(total_ngrams).most_common())
            
        # Removing trigrams if any word is in stop words
        new_dict = {}
        for key,val in total_ngrams_dict.items():
            # checking if any word is in stop word 
            if len([1 for tok in key if tok in stop_words]) == 0:
                new_dict[key] = val

        total_ngrams_dict = new_dict
        
    if n == 4:
        # finding 4grams from data
        #total_ngrams = list(ngrams([word for doc in texts for word in doc.split()],n = n))
        total_ngrams = [list(ngrams(t.split(),n = n)) for t in texts]
        merged = list(itertools.chain(*total_ngrams))
        total_ngrams = merged
        threshold = round(thresh * len(texts))
        # calculating frequency counts of 4grams
        total_ngrams_dict = dict(nltk.FreqDist(total_ngrams).most_common())
            
        # Removing 4grams if any word is in stop words
        new_dict = {}
        for key,val in total_ngrams_dict.items():
            # checking if any word is in stop word 
            if len([1 for tok in key if tok in stop_words]) == 0:
                new_dict[key] = val

        total_ngrams_dict = new_dict

    if n == 5:
        # finding 5grams from data
        #total_ngrams = list(ngrams([word for doc in texts for word in doc.split()],n = n))
        total_ngrams = [list(ngrams(t.split(),n = n)) for t in texts]
        merged = list(itertools.chain(*total_ngrams))
        total_ngrams = merged
        threshold = round(thresh * len(texts))
        # calculating frequency counts of 5grams
        total_ngrams_dict = dict(nltk.FreqDist(total_ngrams).most_common())
            
        # Removing 5grams if any word is in stop words
        new_dict = {}
        for key,val in total_ngrams_dict.items():
            # checking if any word is in stop word 
            if len([1 for tok in key if tok in stop_words]) == 0:
                new_dict[key] = val

        total_ngrams_dict = new_dict
        
    # considering ngrams whose frequency is more than threshold
    if len(total_ngrams_dict) > 0:
        updated_dict = {}
        for key,val in total_ngrams_dict.items():
            if val >= threshold:
                updated_dict[' '.join(key)] = val               
    else:
        updated_dict = {}

    return updated_dict
    
# Naming a cluster where all the data points assigned to the particular cluster is passed as a parameter
def topngram_clustername(text):
    '''
    text : text data as pandas series on which name is calculated. 
    '''
    cluster_name = []
    # finding unigrams, bigrams, trigrams
    unigram1 = n_grams_dict(text, n = 1, thresh = 0.75)
    bigram1 = n_grams_dict(text, n = 2, thresh = 0.65) 
    trigram1 = n_grams_dict(text, n = 3, thresh = 0.6)
    fourgram1 = n_grams_dict(text, n = 4, thresh = 0.55)
    fivegram1 = n_grams_dict(text, n = 5, thresh = 0.5)
    
    if len(unigram1)!=0:
        unigram_word = next(iter(unigram1))
    else: 
        unigram_word = ''
    if len(bigram1)!= 0:
        bigram_word = next(iter(bigram1))
    else:
        bigram_word = ''
    if len(trigram1)!=0:
        trigram_word = next(iter(trigram1))
    else:
        trigram_word = ''
    if len(fourgram1)!= 0:
        fourgram_word = next(iter(fourgram1))
    else:
        fourgram_word = ''
    if len(fivegram1)!=0:
        fivegram_word = next(iter(fivegram1))
    else:
        fivegram_word = '' 
        
    name = ''
    if(unigram_word != ''):
        name = name+unigram_word
    if(bigram_word != ''):
        name = name+"/"+bigram_word
    if(trigram_word != ''):
        name = name+"/"+trigram_word
    if(fourgram_word != ''):
        name = name+"/"+fourgram_word
    if(fivegram_word != ''):
        name = name+"/"+fivegram_word
        
    name = name.lstrip('/')
    
    return name