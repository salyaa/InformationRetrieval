import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import normalize
import re
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk

import numpy as np
import pandas as pd
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer, GermanStemmer, ItalianStemmer, SpanishStemmer, EnglishStemmer, ArabicStemmer

from kiwipiepy import Kiwi  # for Korean

nltk.download('stopwords')

from spacy.lang.ko.stop_words import STOP_WORDS as ko_stop


####################### stemming #################################

# languages without Korean, because Korean is not supported by nltk
def preprocess_text(text, stemmer, lang):
    # delete special characters
    text = re.sub(r'[^\w\s]', ' ', text)  # replace special characters with space
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces with single space
    
    # convert to lowercase
    words = word_tokenize(text, language= get_normalized_language_word(lang))
    
    # stemming
    stemmed_words = [stemmer.stem(word) for word in words]
    
    # reseamble the stemmed words to a sentence
    return ' '.join(stemmed_words)


def stemming(corpus_file_path, lang):
    # initialize stemmer
    nltk.download('punkt')
    nltk.download('punkt_tab')
    # load corpus
    df = pd.read_json(corpus_file_path)
    df = df[df['lang'] == lang]
    # stem
    if lang == 'ko':
        stemmer = Kiwi()
    elif lang == 'ar':
        stemmer = ArabicStemmer()
    else:
        stemmer = SnowballStemmer(get_normalized_language_word(lang))
    df['cleared_text'] = df['text'].apply(lambda x: preprocess_text(x, stemmer, lang))

    df.drop(columns=['text'], inplace=True)

    df.rename(columns={'cleared_text': 'text'}, inplace=True)

    df[['docid', 'text', 'lang']].to_json(f'./Data/corpus.json/corpus_cleaned_{lang}.json', orient='records')
    # rename cleaned_text as text

    return df

def get_normalized_language_word(lang):
    if lang == 'en':
        return 'english'
    elif lang == 'de':
        return 'german'
    elif lang == 'fr':
        return 'french'
    elif lang == 'it':
        return 'italian'
    elif lang == 'es':
        return 'spanish'
    elif lang == 'ar':
        return 'arabic'
    else:
        return 'english'
    
    
####################### tfidf-generate ###########################

# for Italian
from spacy.lang.it.stop_words import STOP_WORDS as it_stop
# for Korean
from spacy.lang.ko.stop_words import STOP_WORDS as ko_stop
# for English
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
# for French
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
# for German
from spacy.lang.de.stop_words import STOP_WORDS as de_stop
# for Spanish
from spacy.lang.es.stop_words import STOP_WORDS as es_stop
# for Arabic
from spacy.lang.ar.stop_words import STOP_WORDS as ar_stop

# choose stopwords based on language
def _get_stopwords(language):
    if language == 'it':  # Italian
        return list(it_stop)
    elif language == 'ko':  # Korean
        return list(ko_stop)
    elif language == 'en':  # English
        return list(en_stop)
    elif language == 'fr':  # French
        return list(fr_stop)
    elif language == 'de':  # German
        return list(de_stop)
    elif language == 'es':  # Spanish
        return list(es_stop)
    elif language == 'ar':  # Arabic
        return list(ar_stop)
    else:
        return []

# load JSON files
def _load_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# extract texts and ids from JSON
def _extract_texts_and_ids_from_json(json_data, language='en', cleared=False):
    # suppose data is stored as the format：{'docid': 'text'}
    texts = []
    doc_ids = []
    for doc in json_data:
        # choose certain language
        if doc['lang'] == language:
            doc_id = doc['docid']
            doc_content = doc['text']
            texts.append(doc_content)  # text
            doc_ids.append(doc_id)     # id
    return doc_ids, texts

# get TF-IDF matrix and vectorizer
def _generate_tfidf_matrix(texts, lang, max_features=5000):
    stopwords = _get_stopwords(lang)
    vectorizer = CountVectorizer(max_features=max_features, stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform(texts)  # vectorize
    doc_freq = np.sum(tfidf_matrix > 0, axis=0) 
    numb_docs = tfidf_matrix.shape[0] 
    idf = np.log((numb_docs + 1) / (doc_freq + 1)) + 1 
    tfidf_matrix = tfidf_matrix.multiply(idf)
    
    return vectorizer, tfidf_matrix

# save vectorizer and corpus
def _save_vectorizer_and_corpus(vectorizer, corpus, vectorizer_path, corpus_path):
    # save vectorizer(used for query)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # save corpus
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for doc in corpus:
            f.write(doc + '\n')

# save doc_id and TF-IDF vector mapping(as the format of list)
# map: {doc_id: [tfidf_vector]}
def _save_docid_tfidf_mapping(doc_ids, tfidf_matrix, tfidf_path,batch_size=1000):
    docid_tfidf_mapping = {doc_id: tfidf_matrix[i].toarray().tolist() for i, doc_id in enumerate(doc_ids)}

    with open(tfidf_path, 'w', encoding='utf-8') as f:
        json.dump(docid_tfidf_mapping, f)   

def _create_language_subfolder(base_path, lang):
    lang_folder = os.path.join(base_path, 'tfidf', lang)
    os.makedirs(lang_folder, exist_ok=True)
    return lang_folder

# generate TF-IDF vectors, vectorizer and corpus
def tfidf_generate(json_file_path, base_path, lang, max_features=10000):
    # Create language-specific subfolder
    lang_folder = _create_language_subfolder(base_path, lang)
    
    # Define file paths
    vectorizer_path = os.path.join(lang_folder, f'saved_vectorizer.pkl')
    corpus_path = os.path.join(lang_folder, f'saved_corpus.txt')
    tfidf_path = os.path.join(lang_folder, f'saved_vectors.json')

    # load JSON file
    json_data = _load_json_data(json_file_path)
    
    # extract
    doc_ids, texts = _extract_texts_and_ids_from_json(json_data, lang)
    
    # get TF-IDF matrix and vectorizer
    vectorizer, tfidf_matrix = _generate_tfidf_matrix(texts, lang, max_features)
    
    # save vectorizer, corpus and doc_id and TF-IDF vector mapping(list)
    _save_vectorizer_and_corpus(vectorizer, texts, vectorizer_path, corpus_path)
    _save_docid_tfidf_mapping(doc_ids, tfidf_matrix, tfidf_path)

    print("successfully saved vectorizer, corpus and doc_id and TF-IDF vector mapping !")


############################ load resources ############################

# load vectorizer and TF-IDF vectors
def load_resources(vectorizer_path, tfidf_vectors_path,chunk_size=1000):
    # vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # TFIDF vectors
    with open(tfidf_vectors_path, 'r', encoding='utf-8') as f:
        tfidf_vectors = json.load(f)
       
    return vectorizer, tfidf_vectors


# transform query to TF-IDF vector
def transform_query_to_tfidf(vectorizer, query_text):
    return vectorizer.transform([query_text])



############################# search #############################

# test query
def query_test( query_data, tfidf_vectors, vectorizer, lang):
    doc_ids = list(tfidf_vectors.keys())
    tfidf_matrix = get_tfidf_matrix_from_vectors(tfidf_vectors)
    my_length = len(query_data[query_data['lang'] == lang])
    idx = np.random.randint(my_length)
    query_example = query_data[query_data['lang'] == lang].iloc[idx]
    query_text = query_example['query']
    query_id = query_example['positive_docs']

    query_vector = transform_query_to_tfidf(vectorizer, query_text)
    top_10_results = get_top_10_similar_docs(tfidf_matrix, query_vector, doc_ids)

    # print result
    print('the query is: ', query_text)
    print('the true document id is: ', query_id)

    # print top 10 results
    for doc_id, similarity in top_10_results:
        print(f"Document ID: {doc_id}, Cosine Similarity: {similarity:.4f}")

# for cleaned_text
def query_test_cleaned_text(query_data, tfidf_vectors, vectorizer, lang):
    doc_ids = list(tfidf_vectors.keys())
    # initialize stemmer
    nltk.download('punkt')
    nltk.download('punkt_tab')
    # load corpus
    # stem
    stemmer = SnowballStemmer(get_normalized_language_word(lang))
    query_data = query_data[query_data['lang'] == lang]
    query_data['text'] = query_data['query'].apply(lambda x: preprocess_text(x, stemmer, lang))
    tfidf_matrix = get_tfidf_matrix_from_vectors(tfidf_vectors)
    my_length = len(query_data[query_data['lang'] == lang])
    idx = np.random.randint(my_length)
    query_example = query_data[query_data['lang'] == lang].iloc[idx]
    query_text = query_example['text']
    query_id = query_example['positive_docs']

    query_vector = transform_query_to_tfidf(vectorizer, query_text)
    top_10_results = get_top_10_similar_docs(tfidf_matrix, query_vector, doc_ids)

    # print result
    print('the query is: ', query_text)
    print('the true document id is: ', query_id)

    # print top 10 results
    for doc_id, similarity in top_10_results:
        print(f"Document ID: {doc_id}, Cosine Similarity: {similarity:.4f}")


def get_top_10_similar_docs(tfidf_matrix, query_vector, doc_ids):
    # normalize the vectors
    tfidf_matrix_normalized = normalize(tfidf_matrix)
    query_vector_normalized = normalize(query_vector.reshape(1, -1))

    # calculate cosine similarity between query_vector and all documents in tfidf_matrix
    cosine_similarities = tfidf_matrix_normalized @ query_vector_normalized.T

    # get top 10 indices
    top_10_indices = np.argsort(-cosine_similarities.flatten())[:10]

    # get top 10 doc_ids and similarities
    top_10_doc_ids = [doc_ids[i] for i in top_10_indices]
    top_10_similarities = cosine_similarities[top_10_indices].flatten()

    return list(zip(top_10_doc_ids, top_10_similarities))


################################ evaluation ############################


# get matrix from tfidf_vectors
def get_tfidf_matrix_from_vectors(tfidf_vectors):
    doc_ids = list(tfidf_vectors.keys())

    return np.array([tfidf_vectors[doc_id][0] for doc_id in doc_ids])


def _get_query_matrix_from_vectors(query_data, vectorizer, lang):
    queries = query_data[query_data['lang']==lang]['query'].tolist()
    query_matrix = vectorizer.transform(queries)

    # extract positive_id and store in an ordered list
    pos_ids = query_data[query_data['lang']==lang]['positive_docs'].tolist()

    return query_matrix, pos_ids

def evaluate_accuracy_from_vectors(query_data, tfidf_vectors, vectorizer, lang):
    doc_ids = list(tfidf_vectors.keys())  # get all document IDs
    # suppose we have tfidf_matrix, query_matrix and pos_ids
    # tfidf_matrix(n_docs x 5000)
    # query_matrix(n_queries x 5000)
    # pos_ids: list of positive document IDs
    # doc_ids: list of all document IDs
    tfidf_matrix = get_tfidf_matrix_from_vectors(tfidf_vectors)  # get tfidf_matrix
    query_matrix, pos_ids = _get_query_matrix_from_vectors(query_data, vectorizer, lang)  # get query_matrix
    # normalize the vectors
    tfidf_matrix_normalized = normalize(tfidf_matrix)
    query_matrix_normalized = normalize(query_matrix)

    # calculate cosine similarity between query_matrix and all documents in tfidf_matrix
    cosine_similarities = query_matrix_normalized @ tfidf_matrix_normalized.T

    # get top 10 indices
    correct_count = 0  # count
    for i in range(cosine_similarities.shape[0]):
        top_10_indices = np.argsort(-cosine_similarities[i])[:10]  # get top 10 indices
        top_10_doc_ids = [doc_ids[j] for j in top_10_indices]  #  get top 10 doc_ids

        # check if positive id is in top 10
        if pos_ids[i] in top_10_doc_ids:
            correct_count += 1  #  if positive id is in top 10, count + 1

    # calculate accuracy
    accuracy = correct_count / len(pos_ids)
    # print accuracy
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def evaluate_accuracy_from_matrix(query_matrix, tfidf_matrix, pos_ids, doc_ids):
    # normalize the vectors
    tfidf_matrix_normalized = normalize(tfidf_matrix)
    query_matrix_normalized = normalize(query_matrix)

    # calculate cosine similarity between query_matrix and all documents in tfidf_matrix
    cosine_similarities = query_matrix_normalized @ tfidf_matrix_normalized.T

    # get top 10 indices
    correct_count = 0  # count
    for i in range(cosine_similarities.shape[0]):
        top_10_indices = np.argsort(-cosine_similarities[i])[:10]  # get top 10 indices
        top_10_doc_ids = [doc_ids[j] for j in top_10_indices]  #  get top 10 doc_ids

        # check if positive id is in top 10
        if pos_ids[i] in top_10_doc_ids:
            correct_count += 1  #  if positive id is in top 10, count + 1

    # calculate accuracy
    accuracy = correct_count / len(pos_ids)
    # print accuracy
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def _get_query_matrix_from_vectors_for_cleaned_text(query_data, vectorizer, lang):
    queries = query_data[query_data['lang']==lang]['text'].tolist()
    query_matrix = vectorizer.transform(queries)

    # extract positive_id and store in an ordered list
    pos_ids = query_data[query_data['lang']==lang]['positive_docs'].tolist()

    return query_matrix, pos_ids



def evaluate_accuracy_from_vectors_for_cleaned_text(query_data, tfidf_vectors, vectorizer, lang):
    doc_ids = list(tfidf_vectors.keys())  # get all document IDs
    
    stemmer = SnowballStemmer(get_normalized_language_word(lang))  # initialize stemmer
    query_data = query_data[query_data['lang'] == lang]
    query_data = query_data.copy()
    query_data['text'] = query_data['query'].apply(lambda x: preprocess_text(x, stemmer, lang))

    tfidf_matrix = get_tfidf_matrix_from_vectors(tfidf_vectors)  # get tfidf_matrix
    query_matrix, pos_ids = _get_query_matrix_from_vectors_for_cleaned_text(query_data, vectorizer, lang)  # get query_matrix
    # normalize the vectors
    tfidf_matrix_normalized = normalize(tfidf_matrix)
    query_matrix_normalized = normalize(query_matrix)

    # calculate cosine similarity between query_matrix and all documents in tfidf_matrix
    cosine_similarities = query_matrix_normalized @ tfidf_matrix_normalized.T

    # get top 10 indices
    correct_count = 0  # count
    for i in range(cosine_similarities.shape[0]):
        top_10_indices = np.argsort(-cosine_similarities[i])[:10]  # get top 10 indices
        top_10_doc_ids = [doc_ids[j] for j in top_10_indices]  #  get top 10 doc_ids

        # check if positive id is in top 10
        if pos_ids[i] in top_10_doc_ids:
            correct_count += 1  #  if positive id is in top 10, count + 1

    # calculate accuracy
    accuracy = correct_count / len(pos_ids)
    # print accuracy
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy
