## [COL 764 Project] Knowledge-based Question Answering using Link Prediction technique

Overview of the Approach Used

<img width="943" alt="Screenshot 2021-11-23 at 11 53 08 PM" src="https://user-images.githubusercontent.com/93838007/143194619-d02d3059-9c2d-423d-9f7c-2ef0cd6aa9d7.png">

## Environment Requirements

```
Python 3.7, PyTorch 1.6
```

### Set data file paths and folder 

1. 

For ```main_train_vocab.py```, provide the following paths in line 124-128.

```
datasetName = "MetaQA"			#dataset folder name
data_path = "./Data/MetaQA/1-hop/vanilla/qa_train.txt"			# training QA pairs path
valid_data_path = "./Data/MetaQA/1-hop/vanilla/qa_dev.txt"		# validation QA pairs path
kbPath ="./Data/MetaQA/kb.txt"									# KG path
```

For ```main_train_roberta.py```, provide the following paths in line 166-170.

```
datasetName = "MetaQA"			#dataset folder name
data_path = "./Data/MetaQA/1-hop/vanilla/qa_train.txt"			# training QA pairs path
valid_data_path = "./Data/MetaQA/1-hop/vanilla/qa_dev.txt"		# validation QA pairs path
kbPath ="./Data/MetaQA/kb.txt"									# KG path
```

2. 

For ```main_test_vocab.py```, provide the following paths in line 129-132.

```
datasetName = "MetaQA"			#dataset folder name
data_path = "./Data/MetaQA/1-hop/vanilla/qa_train.txt"			# training QA pairs path (Used just to calculate vocab size)
test_data_path = "./Data/MetaQA/1-hop/vanilla/qa_test.txt"		# test QA pairs path
```

For ```main_test_roberta.py```, provide the following paths in line 131-133.

```
datasetName = "MetaQA"			#dataset folder name
test_data_path = "./Data/PQ-2/test_q.txt" 						# test QA pairs path
```

3. The ```Models``` folder should be available before running the code.  


## Instructions to run code 
```
1. pip install -r requirements.txt 
2. To train model using LSTM + RoBERTa: python main_train_roberta.py 
3. To train model using only LSTM: python main_train_vocab.py 
4. To test the model using LSTM + RoBERTa: python main_test_roberta.py
5. To test the model using only LSTM: python main_test_vocab.py
```

## References
1.	Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings (ACL 2020). 
2.	Complex Embeddings for Simple Link Prediction (ICML 2016). 
3.	An Interpretable Reasoning Network for Multi-Relation Question Answering (COLING 2018)
