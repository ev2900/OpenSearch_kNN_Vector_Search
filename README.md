# OpenSearch kNN Vector Search

This example uses the publicly avaiable [Amazon Product Question Answer](https://registry.opendata.aws/amazon-pqa/) (PQA) data set. In this example, the questions in the PQA data set are tokenized and represented as vectors. BERT via. Hugging Face is used to generate the embeddings. The vector representation of the questions (embeddings) are loading to an OpenSearch index as a *knn_vector* data type. 

Searches are executed against OpenSearch by transforming search text into embeddings and determining similarity using kNN. The most similar result answers are returned as search results.

# Deployment on AWS 



# How does the Example Python Script Work

## 1. Prepare the headset production question answer (PQA) data

## 2. Convert the question text in the PQA data set into vector(s)

## 3. Create an OpenSearch index

## 4. Load data into the index

## 5. Convert user input/search into a vector

## 6. Search OpenSearch using the vector representation of the user input/search
