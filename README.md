# OpenSearch kNN Vector Search

This example uses the publicly avaiable [Amazon Product Question Answer](https://registry.opendata.aws/amazon-pqa/) (PQA) data set. In this example, the questions in the PQA data set are tokenized and represented as vectors. BERT via. Hugging Face is used to generate the embeddings. The vector representation of the questions (embeddings) are loading to an OpenSearch index as a *knn_vector* data type

Searches are executed against OpenSearch by transforming search text into embeddings and determining similarity using kNN. The most similar result answers are returned as search results

# Deployment on AWS 

To deploy this example on AWS you can click on the button below to launch a CloudFormation stack

[![Launch CloudFormation Stack](https://sharkech-public.s3.amazonaws.com/misc-public/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=open-search-kNN&templateURL=https://sharkech-public.s3.amazonaws.com/misc-public/OpenSearch_kNN_Vector_Search.yaml)

The stack will deploy an Amazon OpenSearch domain and a Cloud9 environment with this GitHub repository downloaded. You can update and run the python script in the Cloud9 environment 

The only parts of the Python script that need to updated before running it is the section below

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

Each JSON document in the raw PQA data set has a question with many potential answers in additon to other information about the product in question 

The code below creates a pandas data frame (df) where each row is a single question and answer pair. The other product information is also removed

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

## 3. Create an OpenSearch index

## 4. Load data into the index

## 5. Convert user input/search into a vector

## 6. Search OpenSearch using the vector representation of the user input/search
