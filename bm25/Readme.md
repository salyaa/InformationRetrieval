# Project Name

This project involves text preprocessing, query handling, and result formatting, with multilingual text data processing and retrieval capabilities.

### Project Test

For the class test for our project, here we preprocess the corpus, but remain the test.csv. So you can run the following codes to test.

We have tested that it only takes 3 minutes to get the results!

```bash
   pip install -r requirements.txt
   python test.py
```

## Project Structure

### Directory Structure

- `Data/`
  - `english/`
    - `corpus_part/`: Contains parts of the English corpus, useful for loading and processing large datasets in segments.
    - `joblibs/`: Stores intermediate results and model files related to the English corpus, such as vectorizers and cached computations.
  - `filtered_corpus/`: Contains filtered corpus files for further processing.
  - `pkl/`: Stores `.pkl` files generated during processing, which may contain serialized intermediate data.
  - `preprocess_corpus/`: Contains intermediate files for corpus preprocessing.
  - `preprocess_query/`: Contains files related to query preprocessing, specifically for handling query-related data.
  - `preprocess_test/`: Contains files for preprocessing test data used to validate preprocessing flows.

### File Descriptions

- `english_work.ipynb`: Jupyter Notebook for processing and analyzing the English corpus, including text processing and retrieval operations.
- `helpers.py`: Python script with helper functions, likely including functions for preprocessing, formatting, or retrieval tasks.
- `preprocess.ipynb`: Jupyter Notebook for preprocessing, including cleaning, tokenization, and other steps for the corpus and query data.
- `requirements.txt`: Lists the required Python packages and dependencies for the project, useful for setting up the environment.
- `submission.csv`: Final generated query results file, suitable for submission or retrieval evaluation.
- `test.csv`: Test dataset file used to validate retrieval and formatting processes.
- `test.py`: Test script to run and validate the functionality of various modules.
- `work_with_pkl.ipynb`: Jupyter Notebook for data processing and analysis that uses `.pkl` files.
- `work_without_pkl.ipynb`: Jupyter Notebook for data processing that does not rely on `.pkl` files.

## Usage

1. Clone the project and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. run test.py
    ```bash
    python test.py
    ```

### Notes
Ensure that required nltk and spacy datasets are downloaded. You can add download commands in the code to make sure dependencies are available.

For large files and datasets, place them in the corresponding folders (e.g., corpus_part and pkl directories).

### Contribution
Contributions to improve the code, fix bugs, or add new features are welcome. Please submit a Pull Request or open an Issue to discuss potential changes.