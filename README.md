# OpenSearch kNN Vector Search

<img width="275" alt="map-user" src="https://img.shields.io/badge/cloudformation template deployments-16-blue"> <img width="85" alt="map-user" src="https://img.shields.io/badge/views-1703-green"> <img width="125" alt="map-user" src="https://img.shields.io/badge/unique visits-567-green">

This example uses the publicly avaiable [Amazon Product Question Answer](https://registry.opendata.aws/amazon-pqa/) (PQA) data set. In this example, the questions in the PQA data set are tokenized and represented as vectors. BERT via. Hugging Face is used to generate the embeddings. The vector representation of the questions (embeddings) are loading to an OpenSearch index as a *knn_vector* data type

Searches are executed against OpenSearch by transforming search text into embeddings and determining similarity using kNN. The most similar result answers are returned as search results

# Deployment on AWS

To deploy this example on AWS you can click on the button below to launch a CloudFormation stack

[![Launch CloudFormation Stack](https://sharkech-public.s3.amazonaws.com/misc-public/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=open-search-kNN&templateURL=https://sharkech-public.s3.amazonaws.com/misc-public/OpenSearch_kNN_Vector_Search.yaml)

The stack will deploy an Amazon OpenSearch domain and a Cloud9 environment with this GitHub repository downloaded. Before using the Cloud9 enviorment run the [resize_EBS.sh](https://github.com/ev2900/OpenSearch_kNN_Vector_Search/blob/main/resize_EBS.sh) from the Cloud9 termial.

Execute the following in terminal from the *OpenSearch_kNN_Vector_Search* directory

```bash resize_EBS.sh```

The bash script resizes the EBS volume attached to the Cloud9 instance from 10 GB to 100 GB.

Once the resize is complete, uou can update and run the [kNN.py](https://github.com/ev2900/OpenSearch_kNN_Vector_Search/blob/main/kNN.py) python script in the Cloud9 environment. The only parts of the Python script that need to updated before running it is the section below

```
# Configure re-usable variables for Opensearch domain URL, user name and password
opensearch_url = 'https://<opensearch_domain_url' # DO NOT INCLUDE TRAILING SLASH
opensearch_user_name = '<user_name>'
opensearch_password = '<password>'
```

The user name and password are part of the CloudFormation outputs. The domain URL can be found on the [OpenSearch page](https://us-east-1.console.aws.amazon.com/aos/home) in the AWS console under the domains section. Once you update these values in the [kNN.py](https://github.com/ev2900/OpenSearch_kNN_Vector_Search/blob/main/kNN.py) file make sure to save the file.

Install the required Python libraries by running in the Cloud9 terminal

```pip install -r requirements.txt```

Once the required libraries are install run the python script by executing

```python kNN.py```

# How does the Example Python Script Work

This section explains how the python script [kNN.py](https://github.com/ev2900/OpenSearch_kNN_Vector_Search/blob/main/kNN.py) works. The script has 6 sections each section in the full [kNN.py](https://github.com/ev2900/OpenSearch_kNN_Vector_Search/blob/main/kNN.py) script is clearly defined by comments. Each is explained below

## 1. Prepare the headset production question answer (PQA) data

Each JSON document in the raw PQA data set has a question with many potential answers in additon to other information about the product in question. The code below creates a pandas data frame (df) where each row is a single question and answer pair. The other product information is also removed.

For example a JSON document from the raw PQA data set is below

```
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
```
After processing the document the df data frame will have a question and answer column

	Question: 	does this work with cisco ip phone 7942
	Answer: 	Use the Plantronics compatibility guide to see what is compatible with your phone. http://www.plantronics.com/us/compatibility-guide/

## 2. Convert the question text in the PQA data set into vector(s)

After preparing the PQA dataset, we need to tokenize the question text and convert it into a vector representation. We do this using BERT via. Hugging Face. This process has 3 steps. Each is explained below

### Tokenize the question text

Input:  ```df["question"].tolist()``` <br>
Output: ```inputs_tokens```

tokenizer()
	padding - Ensure that all sequences in a batch have the same length. If the padding argument is set to True, the function will pad sequences up to the length of the longest sequence in the batch
	return_tensors - Return output as a PyTorch torch.Tensor object

### Convert tokenized questions into vectors using BERT

Input:  ```inputs_tokens``` <br>
Output: ```outputs```

```outputs``` is 3 dimensional tensor object. Working with 1000 rows of data the dimension of outputs could be [1000, 64, 768]

### Use mean pooling to condense the

Input: ```outputs``` <br>
Ouput: ```question_text_embeddings```

```question_text_embeddings``` is a 2 dimensional tensor object. Working with 1000 rows of data the dimension of output could be [1000, 768]

## 3. Create an OpenSearch index

Make an API call to the OpenSearch domain to create an OpenSearch index named ```nlp_pqa``` with 3 fields. These fields include

1. question_vector
2. question
3. answer

The data type of the ```question_vector``` field is ```knn_vector```

## 4. Load data into the index

Make API calls to the OpenSearch domain to load the data (plain text and vector representation) into the OpenSearch index that was just created

## 5. Convert user input/search into a vector

Tokenize and convert the user input / search of *does this work with xbox?* into a vector. The vector representation of this search will be used in the next step

Input: ```query_raw_sentences = ['does this work with xbox?']```
Ouput: ```search_vector```

## 6. Search OpenSearch using the vector representation of the user input/search

Make an API call to the OpenSearch domain to run the run the search *does this work with xbox* by passing the vector-ized version of the search to OpenSearch. Print the top results to the console
