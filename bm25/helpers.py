import os
import pandas as pd
import numpy as np
import string
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer, GermanStemmer, ItalianStemmer, SpanishStemmer, EnglishStemmer, ArabicStemmer
from kiwipiepy import Kiwi  # for Korean
from spacy.lang.ko.stop_words import STOP_WORDS as ko_stop

nltk.download('stopwords')
nltk.download('punkt')

##################################  spliting  ##################################################

def split_test_by_language(test_file):
    """
    Split the test CSV by language and save each language-specific subset in a separate file.

    Args:
        test_file (str): Path to the main test CSV file.
    """
    # create output folder
    output_folder = "test"
    os.makedirs(output_folder, exist_ok=True)
    
    # read the test file
    try:
        test_df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"File {test_file} not found.")
        return

    # split by language
    languages = test_df['lang'].unique()
    for lang in languages:
        lang_df = test_df[test_df['lang'] == lang]
        lang_file_path = os.path.join(output_folder, f"{lang}_test.csv")
        lang_df.to_csv(lang_file_path, index=False)
        print(f"Saved {lang} language data to {lang_file_path}")


##################################  tokenizing  ##################################################

# check if the output folder exists
output_folder = "test"
os.makedirs(output_folder, exist_ok=True)

# define a function to get the stemmer and stopwords for a given language
def get_stemmer_and_stopwords(language):
    """return the stemmer and stopwords for the specified language"""
    if language == 'en':
        return EnglishStemmer(), stopwords.words("english")
    elif language == 'fr':
        return FrenchStemmer(), stopwords.words("french")
    elif language == 'de':
        return GermanStemmer(), stopwords.words("german")
    elif language == 'it':
        return ItalianStemmer(), stopwords.words("italian")
    elif language == 'es':
        return SpanishStemmer(), stopwords.words("spanish")
    elif language == 'ar':
        return ArabicStemmer(), stopwords.words("arabic")
    elif language == 'ko':
        return Kiwi(), list(ko_stop)
    else:
        raise ValueError(f"Unsupported language: {language}")

# define a function to process and tokenize text based on the language
def process_text(text, stemmer, stop_words, language):
    """tokenize the text based on the language"""
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    if language != 'ko':
        tokens = word_tokenize(text) 
        tokenized_words = [stemmer.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    else:
        tokens = stemmer.analyze(text)[0][0] 
        tokenized_words = [word.form for word in tokens if word.tag.startswith('N') and word.form not in stop_words] 
    
    return tokenized_words

# define a function to tokenize a given column in a DataFrame
def tokenizer(df, col_name, language, mytype):
    """tokenize the text in the specified column of the DataFrame"""
    stemmer, stop_words = get_stemmer_and_stopwords(language)
    
    new_df = df.copy()
    new_col_name = col_name + "_token"
    new_df[new_col_name] = new_df[col_name].apply(lambda text: process_text(text, stemmer, stop_words, language))
    
    print(f"Data has been successfully tokenized for language: {language}")
    
    # save the new DataFrame as a CSV file
    output_path = os.path.join(output_folder, f"bm25_{language}_{mytype}.csv")
    new_df.to_csv(output_path, index=False)
    print(f"New data saved as CSV at {output_path}!")
    
    return new_df

# define a function to process all languages
def process_all_languages(languages):
    for lang in languages:
        lang_file = os.path.join(output_folder, f"{lang}_test.csv")
        if os.path.exists(lang_file):
            df = pd.read_csv(lang_file)
            tokenizer(df, "query", lang, "test")  # tokenize the query column
        else:
            print(f"File for language {lang} not found at {lang_file}.")




#############################################  BM25  ####################################################


# query for BM25
def bm25_score(queries, corpus_file):
    # load the corpus
    idf_times_F_adjusted, vectorizer, doc_ids = joblib.load(corpus_file)
    
    # load the queries
    query_term_matrix = vectorizer.transform(queries['query_token'])
    
    # calculate the BM25 scores
    BM25_scores = query_term_matrix @ idf_times_F_adjusted.T
    
    # get the top 10 documents for each query
    pos_docs = {}
    for query_idx, query_id in enumerate(queries['query_id']):
        scores = BM25_scores[query_idx]
        top_doc_indices = np.argsort(scores)[-10:][::-1]
        pos_docs[query_id] = [doc_ids[idx] for idx in top_doc_indices]
    
    return pos_docs

# english, with split
def _bm25_query_part(query, part_file, vectorizer, top_n=10):
    idf_times_F_adjusted, doc_ids = joblib.load(part_file)
    idf_times_F_adjusted = idf_times_F_adjusted.toarray()
    query_term_matrix = vectorizer.transform(query['query_token'])
    BM25_scores = query_term_matrix @ idf_times_F_adjusted.T
    
    pos_docs = []
    for query_idx in range(query_term_matrix.shape[0]):
        scores = BM25_scores[query_idx]
        top_doc_indices = np.argsort(scores)[-top_n:][::-1]
        pos_docs.extend([(doc_ids[idx], scores[idx]) for idx in top_doc_indices])
    
    return pos_docs

def bm25_query(query_file, vectorizer, n_splits=10, top_n=10):
    queries = pd.read_csv(query_file)
    all_results = {query_id: [] for query_id in queries['query_id']}
    for i in range(n_splits):
        part_file = f'Data/english/joblibs/corpus_part_{i}_bm25.joblib'
        part_results = _bm25_query_part(queries, part_file, vectorizer, top_n=top_n)
        
        for query_idx, query_id in enumerate(queries['query_id']):
            all_results[query_id].extend(part_results[query_idx * top_n:(query_idx + 1) * top_n])

    final_results = {}
    for query_id, docs in all_results.items():
        sorted_docs = sorted(docs, key=lambda x: x[1], reverse=True)[:top_n]
        final_results[query_id] = [doc_id for doc_id, score in sorted_docs]

    return final_results







#############################################  Storing  ####################################################
def format_and_combine(language_order, input_folder, output_file):
    """
    Combine BM25 results for different languages and format the output.

    Parameters:
    language_order (list): list of languages in the order they should be combined
    input_folder (str): folder containing BM25 results for each language
    output_file (str): path to save the formatted output
    """
    # start with an empty DataFrame
    combined_df = pd.DataFrame()

    # loop through each language and read the BM25 results
    for lang in language_order:
        file_path = os.path.join(input_folder, f"bm25_results_{lang}.csv")
        lang_df = pd.read_csv(file_path)

        # check if the required columns are present
        if 'doc_ids' not in lang_df.columns:
            raise ValueError(f"file {file_path} does not contain 'doc_ids' column")

        # only keep the 'doc_ids' column and add it to the combined DataFrame
        combined_df = pd.concat([combined_df, lang_df[['doc_ids']]], ignore_index=True)

    # add an 'id' column to the combined DataFrame
    combined_df.insert(0, 'id', range(0, len(combined_df) ))

    # rename the 'doc_ids' column to 'docids
    combined_df = combined_df.rename(columns={"doc_ids": "docids"})

    # convert the 'docids' column to string
    combined_df['docids'] = combined_df['docids'].apply(lambda x: str(x.strip().split()))

    # save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"saving the combined results to {output_file}")

    # return the combined DataFrame
    return combined_df