from helpers import *

# set the path to the test file
languages = ['en', 'fr', 'de', 'it', 'es', 'ar', 'ko']
test_file = 'test.csv'
split_test_by_language(test_file)
process_all_languages(languages)

# for non-English languages
languages = ['fr', 'de', 'it', 'es', 'ar', 'ko']

# bm25 corpus
for lang in languages + ['en']:
    if lang == 'en':
        # english
        vectorizer = joblib.load(f'Data/english/joblibs/vectorizer_en.joblib')
        query_file = 'test/bm25_en_test.csv'
        pos_docs = bm25_query(query_file, vectorizer)
    else:
        # non-english
        corpus_path = f"Data/pkl/bm25_corpus_{lang}.pkl"
        query_file = f"test/bm25_{lang}_test.csv"
        
        if os.path.exists(corpus_path) and os.path.exists(query_file):
            queries = pd.read_csv(query_file)
            pos_docs = bm25_score(queries, corpus_path)
        else:
            print(f"Corpus or query file for language {lang} not found.")
            continue

    # save the results
    
    # check if the output folder exists
    output_folder = "test/result"
    os.makedirs(output_folder, exist_ok=True)
    output_file = f"test/result/bm25_results_{lang}.csv"
    with open(output_file, "w") as f:
        f.write("query_id,doc_ids\n")
        for query_id, doc_list in pos_docs.items():
            f.write(f"{query_id},{' '.join(doc_list)}\n")
    print(f"Results saved for language {lang} at {output_file}")


# set arguments
language_order = ['en', 'fr', 'de', 'es', 'it', 'ko', 'ar']
input_folder = "test/result"
output_file = "submission.csv"

result_df = format_and_combine(language_order, input_folder, output_file)

# print the first few rows of the result
print("\nCombined results:")
print(result_df.head())