
# The Document Vectors Using Cosine Similarity Revisited
This repository contains code for the paper under the same name.

Code in folder "src" and file "ensemble.py" are from [Thongtan's project](https://github.com/tanthongtan/dv-cosine). 

## Download and prepare the datasets
### Document embedding on IMDB review dataset
Run

    python download.py

to download needed files from [Thongtan's project](https://github.com/tanthongtan/dv-cosine)

Run 

    python IMDB_splits/collect_and_download.py
    python IMDB_splits/gather_all.py

to download and process the splits of the IMDB datset used in [previous work](https://github.com/allenai/dont-stop-pretraining)s.

## Re-evaluation of Document Vectors using Cosine Similarity
### Document matching between 1gram and 3gram

In the work of DV-ngrams-cosine, a preprocessed dataset instead of the original IMDB review dataset was used. The preprocessed dataset consists of three files, for unigrams (1gram), unigrams + bigrams, unigrams + bigrams + trigrams (3gram), respectively. Noticeably, 1gram and 3gram have different orders of documents. In the ensemble, DV-ngrams-cosine vectors were trained with 3gram, while BON vectors were extracted from 1gram, thus the mismatching problem. We restored the correct order by solving this sorting problem with counting sort: argsort(3gram, order=1gram).

First, documents from both 1gram and 3gram were processed so that each document from 1gram should be exactly the same as the corresponding document from 3gram. Then the proposed sorting problem was solved by counting sort with a hash table. The retrieved order was then checked to reassure that it was a real permutation of the original order (no repeats).

### Restoring the preprocessing method

To ensure the preprocessed files (1gram and 3gram) were indeed from the original IMDB review dataset, and to enable further works of ours that can directly compare with the original work, the preprocessing method for 1gram was restored. A preprocessing method was proposed and then tested, by directly comparing the documents in 1gram and the preprocessed documents from the original IMDB review dataset with the proposed preprocessing method.

Run

    python original_to_1gram.py

to reproduce the experiments in this part. It will output "matched" if everything went well, otherwise it will throw an assertion error. Check the function "normalize_text" in original_to_1gram.py for details of this preprocessing method.

### Evaluate the ensemble

The ensemble was evaluated by the original code, with both the original and the correct DV-BON matching. Some additional tests with shuffed matching were also carried out. The shuffing schemes are as shown the following table. Embeddings were repeatedly read and written to new files to enable the tests with the original code `ensemble.py`. All tests involving shuffing were run for 30 times to better estimate the scores.

||in-class|cross-class|
|----------------------------------|:--:|:--:|
|test set                          |A|B|
|train and test sets (respectively)|C|D|

Run

    python test_with_origin.py

to reproduce the experiments for the first 2 parts (restoring the correct order and evaluate the ensemble). Find the log and report in folder "test_logs"

Our results are shown in the following table

||Score Mean|Score Std|
|----------------------------------|:-----:|:-----:|
|original matching| 97.42||
|correct matching| 93.68||
|test set shuffed in-class (A)| 96.58| 0.07|
|test set shuffed cross-class (B)| 61.80| 0.25|
|train/test shuffed in-class (C)| 97.43| 0.08|
|train/test shuffed cross-class (D)| 91.64| 0.08|

Comparing A and B shows that when test set is shuffled dis-respecting the classes, the score dropped very significantly, which means the embedding is important in the ensemble. C reproduced very similar scores as the original matching, implying contribution of data leakage to the high score. D serves as a control group to C.


### Document Vectors using Cosine Simialrity + BON: gridsearch

run 

    python ensemble_gs.py
    
to run the grid-search, and then refer to dv+bon_results.ipynb for results.

### Document Vectors using Cosine Similarity + RoBERTa: gridsearch

run 

    python roberta/roberta.py

to finetune the RoBERTa model and extract embeddings for the documents.

Then check out dv+roberta.ipynb for additional experiments and results.

## Training Set Size Experiment

Compare DV-ngrams-cosine, BON, DV-ngrams-cosine + BON and RoBERTa with different training sizes.

run 

    python imdb_run1.py

to train DV-ngrams-cosine (**This requires a working java installation**), and then run `imdb_trainsize_experiment.ipynb` and `trainsize_roberta.py` to finish the experiments.

Then check `imdb_trainsize_experiment_sum.ipynb` for results.


## Naive Bayesian Sub-sampling

run

    python imdb_runs.py

to complete all the runs of embedding training. **This requires a working java installation**.

Use `nb_vs_full.ipynb` to check out all the results and plots.


## Appendix B.3

Refer to `compare_2_ways_of_concatenation.ipynb`.



