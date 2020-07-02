# PT-BR Toxic Tweets Classification

Repository for [Toxic Language Detection in Social Media for Brazilian Portuguese: New Dataset and Multilingual Analysis]().

## Prerequisites
Python Version: 3.8.3

To install required libraries:
```
python -m pip install -r requirements.txt
```

### Setting up Files
* Unzip all the toxic tweets raw files into ```data/raw_data/toxic```
* Unzip all the general (non-keyword based) tweets raw files into ```data/raw_data/generic```

### Generate Dataset
```
python generate_dataset.py --max_tweets INT --tweet_type STR
```

### Generate Embeddings
```
python generate_embeddings --max_tweets INT --tweet_type STR --window_size INT --vector_dim INT
```

### Run the Experiments
* zip the folder ```experiments/datasets```
* Create a [google drive](https://drive.google.com/) folder and unzip the file into it
* Upload ```experiments/toxic_classification.ipynb``` into this folder
* Run the desired section

## ToLD-BR
told-br_sample.zip contains 100 randomly selected tweets from our dataset and a readme with more details.
