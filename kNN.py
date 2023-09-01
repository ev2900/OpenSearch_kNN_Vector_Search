import torch

import boto3
import re
import time
import sagemaker

import json
import pandas as pd

import requests

from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, DistilBertModel

# 1. Prepare headset production question answer (PQA) data
'''
Each JSON document in the raw PQA data set has a question with many potential answers in additon to other information about the product in question. 

The code below creates a pandas data frame (df) where each row is a single question and answer pair. The other product information is also removed.

For example a JSON document from the raw PQA data set is below 

	{
		"question_id": "Tx39GCUOS5AYAFK",
		"question_text": "does this work with cisco ip phone 7942",
		"asin": "B000LSZ2D6",
		"bullet_point1": "Noise-Canceling microphone filters out background sound",
		"bullet_point2": "HW251N P/N 75100-06",
		"bullet_point3": "Uses Plantronics QD Quick Disconnect Connector. Must be used with Plantronics Amp or with proper phone or USB adapter cable",
		"bullet_point4": "Connectivity Technology: Wired, Earpiece Design: Over-the-head, Earpiece Type: Monaural, Host Interface: Proprietary, Microphone Design: Boom, Microphone Technology: Noise Canceling, Product Model: HW251N, Product Series: SupraPlus, Standard Warranty: 2 Year",
		"bullet_point5": "Easy Lightweight Wear -Leaving One Ear Uncovered For Person-to-Person Conversations", "product_description": "", "brand_name": "Plantronics", "item_name": "Plantronics HW251N SupraPlus Wideband Headset (64338-31)",
		"question_type": "yes-no",
		"answer_aggregated": "neutral",
		"answers": [
			{"answer_text": "Use the Plantronics compatibility guide to see what is compatible with your phone. http://www.plantronics.com/us/compatibility-guide/"},
			{"answer_text": "I think that you will need a extra cord, but, To avoid offering you any false information,   We highly recommend contacting the manufacturer of this product for more specific information.   if you are not sure about it, you can go first to :  http://www.plantronics.com/us/support/  or call Plantronics TOLL FREE SUPPORT: 1-855-765-7878 24-HOUR SUPPORT SUNDAY 2PM-FRIDAY 5PM (PT)  they will answer all the questions you need to know about it."},
			{"answer_text": "I'm really not positive. It works with our phones that include model numbers 7941, 7945 and 7961."}
		]
	}

After processing the document the df data frame will have a question and answer column

	Question: 	does this work with cisco ip phone 7942
	Answer: 	Use the Plantronics compatibility guide to see what is compatible with your phone. http://www.plantronics.com/us/compatibility-guide/

'''

print("Preparing data set")

number_of_rows_from_dataset = 1000

df = pd.DataFrame(columns=('question', 'answer'))

with open('amazon-pqa/amazon_pqa_headsets.json') as f:
    i=0
    for line in f:
        data = json.loads(line)
        df.loc[i] = [data['question_text'],data['answers'][0]['answer_text']]
        i+=1
        # optional 
        if(i == number_of_rows_from_dataset):
            break

# 2. Convert the question text in the PQA data set into vector(s)
'''
Step 1. Tokenize the question text

	Input:  df["question"].tolist()
	Output: inputs_tokens

	tokenizer()
		padding - Ensure that all sequences in a batch have the same length. If the padding argument is set to True, the function will pad sequences up to the length of the longest sequence in the batch
		return_tensors - Return output as a PyTorch torch.Tensor object

Step 2. Convert tokenized questions into vectors using BERT

	Input:  inputs_tokens
	Output: outputs

	outputs is 3 dimensional tensor object. Working with 1000 rows of data the dimension of outputs could be [1000, 64, 768]

Step 3. Use mean pooling to condense the 

	Input: outputs
	Ouput: question_text_embeddings

	question_text_embeddings is a 2 dimensional tensor object. Working with 1000 rows of data the dimension of output could be [1000, 768]

'''

# Tokenize the questions in the PQA data set
print("Tokenizing the text")

tokenizer = DistilBertTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")

inputs_tokens = tokenizer(df["question"].tolist(), padding=True, return_tensors="pt")

# Convert the tokenized questions into vectors
print("Converting tokenized text into vectors")

model = DistilBertModel.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")

# disable gradient computation to speed up the vector creation
with torch.no_grad():
	outputs = model(**inputs_tokens)

#print('outputs: ' + str(outputs[0].size()))

# Mean pooling
print("Applying mean pooling to vector representation of the text")

token_embeddings = outputs[0] # first element of model_output contains all token embeddings
input_mask_expanded = inputs_tokens['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

question_text_embeddings = sum_embeddings / sum_mask

#print('question_text_embeddings: ' + str(question_text_embeddings.size()))

#print(df["question"])
#print(question_text_embeddings)

# 3. Create an OpenSearch index
'''
Create an OpenSearch index named nlp_pqa with 3 fields. These fields include

	1. question_vector
	2. question
	3. answer

The data type of the question_vector field is knn_vector
'''

print("Creating the OpenSearch index")

# Configure re-usable variables for Opensearch domain URL, user name and password
opensearch_url = 'https://<opensearch_domain_url' # DO NOT INCLUDE TRAILING SLASH
opensearch_user_name = '<user_name>'
opensearch_password = '<password>'

create_knn_index_request_body = {
    "settings": {
        "index.knn": True,
        "index.knn.space_type": "cosinesimil",
        "analysis": {
          "analyzer": {
            "default": {
              "type": "standard",
              "stopwords": "_english_"
            }
          }
        }
    },
    "mappings": {
        "properties": {
            "question_vector": {
                "type": "knn_vector",
                "dimension": 768,
                "store": True
            },
            "question": {
                "type": "text",
                "store": True
            },
            "answer": {
                "type": "text",
                "store": True
            }
        }
    }
}

create_index_r = requests.put(opensearch_url + '/nlp_pqa', auth=(opensearch_user_name, opensearch_password), headers= {'Content-type': 'application/json'}, data=json.dumps(create_knn_index_request_body))
#print(create_index_r.text)

# 4. Load data into the index
'''
Load data into the OpenSearch index that was just created.
'''

print("Loading data to the OpenSearch index")

i = 0
for c in df["question"].tolist():
    question_text_i = c
    question_vector_i = question_text_embeddings[i].tolist()
    answer_i = df["answer"][i]

    #print('Question Text: ' + question_text)
    #print('Question Vector: ' + str(question_vector[0]) + (' ...'))
    #print('Answer: ' + answer)

    upload_document_request_body = {
    	"question_vector": question_vector_i,
    	"question": question_text_i,
    	"answer": answer_i
    }

    upload_document_r = requests.post(opensearch_url + '/nlp_pqa/_doc', auth=(opensearch_user_name, opensearch_password), headers= {'Content-type': 'application/json'}, data=json.dumps(upload_document_request_body))

    #print(upload_document_r.text)

    i+=1

# 5. Convert user input/search into a vector
'''
Conver user input/search in a vector

	Output: search_vector

'''

print("Converting search into a vector")

# ? Refactor this block of code

query_raw_sentences = ['does this work with xbox?']

tokenizer = DistilBertTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
model = DistilBertModel.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
inputs_tokens = tokenizer(query_raw_sentences, padding=True, return_tensors="pt")
    
with torch.no_grad():
    outputs = model(**inputs_tokens)

print("Applying mean pooling to vector representation of the search")

token_embeddings = outputs[0] # first element of model_output contains all token embeddings
input_mask_expanded = inputs_tokens['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

sentence_embeddings = sum_embeddings / sum_mask

search_vector = sentence_embeddings[0].tolist()
# ? 

# 6. Search OpenSearch using the vector representation of the user input/search
'''
Make an API call to run the search using the search_vector created in the last step
'''

print("Search OpenSearch using the vector representation of the search")

query = {
    "size": 30,
    "query": {
        "knn": {
            "question_vector":{
                "vector":search_vector,
                "k":30
            }
        }
    }
}

query_r = requests.get(opensearch_url + '/nlp_pqa/_search', auth=(opensearch_user_name, opensearch_password), headers= {'Content-type': 'application/json'}, data=json.dumps(query))

#print(query_r.text)

# Print search results
json_res = query_r.json()

number_of_results_to_print = 3
results_printed = 0

print('Search results:')

for hit in json_res["hits"]["hits"]:
	if number_of_results_to_print > results_printed:
		print(' ')
		print('Score: ' + str(hit["_score"]))
		print('Answer: ' + hit["_source"]["answer"])
		print('Question: ' + hit["_source"]["question"])
		print(' ')

		results_printed = results_printed + 1
	else:
		break
