README

In the folder provided, the report for our group can be found alongside the code for the three models that we attempted to implement. The three models (bm25, tfidf, query expansion) can be found in their respective folders, with the bm25 model being the final model that we implemented and which can also be found on Kaggle. There are additional files in the bm25 folder which are used to produce some of the data that is on Kaggle, for example the pkl files.

Data
With respect to data, the bm25 model is the one that requires the most preprocessed data in order to run, however since it is the final model, all the data required is visible on Kaggle. In brief, the bm25 model uses pkl files for the corpus of all languages except English, while for English the corpus is split into 10 different parts and saved as a joblib file. This is also true for the vectorizer in English, which is also stored as a joblib. All the queries and the test file are preprocessed and saved as csv files.
For the tfidf model, the data required is a json file of the corpus for each language that has already been tokenized and stemmed, while the query file is the dev.csv file given to us originally. In terms of the query expansion model, the data required are csv files for each language s corpus and query that have been tokenized and stemmed in a method that is similar to the one used for bm25.
