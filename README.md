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

This section explains how the python script [kNN.py](https://github.com/ev2900/OpenSearch_kNN_Vector_Search/blob/main/kNN.py) works. The script has 6 sections. Each is explained below

## 1. Prepare the headset production question answer (PQA) data

## 2. Convert the question text in the PQA data set into vector(s)

## 3. Create an OpenSearch index

## 4. Load data into the index

## 5. Convert user input/search into a vector

## 6. Search OpenSearch using the vector representation of the user input/search
