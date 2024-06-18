# jamm-algorithm

current useful steps:
1. Go to https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts
2. unzip this file and save it as arxiv_data.csv
3. make a virtual environment and `pip install pandas`
4. run `python3 src/preprocess.py ./arxiv_data.csv`. This will create a `arxiv_data.processed.csv`
5. clone https://github.com/maujim/paper-reviewer-matcher
6. from there, run `python mindmatch_cluster.py ./path/to/arxiv_data.processed.csv --n_match=6 --n_trim=50 --n_clusters=4 --output=arxiv-match.csv`
