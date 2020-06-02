# HinDroid-with-Embeddings

![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/davidzz/hindroid-xl)

## Overview

Malware detection for android applications is an expanding field with the introduction of the [Hindroid](https://www.cse.ust.hk/~yqsong/papers/2017-KDD-HINDROID.pdf) model (DOI:[10.1145/3097983.3098026](https://doi.org/10.1145/3097983.3098026)). It proposes a method that transforms the semantic relationships between Android applications and their decompiled source code to a Heterogeneous Information Network (HIN) and uses similarities from various meta-paths between apps to construct a model for malware classification. To further explore the field, we aim to extend the HinDroid model to improve the accuracy in specific subsets of the AMD dataset. Our effort will be focused on finding better representations for both apps as well as APIs and discovering methods to incorporate them as additional features in a new model. In the meantime, we plan to evaluate how the proposed model captures the features that are relevant to the classification task and compare to that of the HinDroid baseline. Our contributions can be utilized in systems where the analysis of malware and interpretable features are more important than mere detection.

## Usage
Docker image on Docker Hub:
[davidzyx/hindroid-xl](https://hub.docker.com/repository/docker/davidzz/hindroid)

On Datahub, create a pod using a custom image with 4 CPU and 32 GB of memory. If you use another configuration, use at least 6GB memory per CPU.
```bash
launch.sh -i davidzz/hindroid-xl -c 4 -m 32
```

Modify the config file located in `config/data-params.json`. If you want to run a test drive, use `config/test-params.json`. Put either `data` or `test` as the first argument.

The HinDroid baseline uses the driver file `run.py` with 3 targets: `ingest`, `process`, and `model`. Put each target space-separated as arguments in the call. To run the whole pipeline, use
```bash
python run.py data ingest process model
```

`process` target will save `.npz` files in `data/processed/` for generating various embeddings.

## System Prerequisites and Definitions

- APK - Executable file for Android
- Smali code - Human readable code decompiled from Dalvik bytecode contained the APK
- Heterogeneous Information Network (HIN) - A graph where its nodes may not be of the same type
- API Extraction
  - Use regex to match specific patterns
  - API calls and method blocks
![api](https://i.imgur.com/nx3rKKv.png)

## HinDroid Efficiency Improvements

In HinDroid, the amount of APIs that are used in the final gram matrix calculations can have dimension of several millions. Even in sparse format, these matrices take up huge computational resources for calculation. We are able to reduce the number of API used in HinDroid to 1000 APIs while retaining the same level of accuracy. Both the time and space complexity for training and inference can be improved by a few orders of magnitude. We achieve this by selecting the top important (larger absolute coefficients) APIs (words) from fitting a logistic regression on each app (document) after applying BM25 extraction on the counts of each API for each app.

| Method            | # of Apps | # of APIs used | RAM used | train+test time |
|-------------------|-----------|----------------|----------|-----------------|
| HinDroid-original | 1670      | 2024313        | 68GB     | 3h29m41s        |
| HinDroid-reduced  | 1670      | 1000           | 0.3GB    | 20s             |

## Embedding Techniques Explored

As we are using NLP approaches such as word2vec which cannot be directly applied to application source code and matrices from HinDroid, our data ingestion pipeline generates text corpus by traversing the graphs following an user defined metapaths and the length of a random walk. Using a metapath `ABPBA` with random walk length 5000, the text corpus may look like 

`app_3 -> api_500 -> api_321 -> api_234 -> api_578 -> app_321 -> api_123…`

where each token represents an application or api node and the neighbouring tokens are
connected by edges in the graph.

### Word2Vec

In a graph where there are two types of nodes: application and api, Word2Vec is the first approach that we attempt to capture the relationship beyond application and apis that have a direct connection in the graph. This traditional and powerful NLP embeddings techniques helps us to learn the similarity between applications not just limited to the shared api, and also the ability to identify the clusters connection between application and api that do not always have a direct connection. Using gensim’s word2vec model, we are able to generate vector embeddings for each application and api found in the text corpus. We successfully converted decompiled Android source code into a vector of numbers for each application and this information can be easily used in a machine learning model.

To evaluate the effectiveness of the generated embeddings, we visualize the embedding clusters by applying dimensionality reduction into two dimensional vectors. The embedding visualization for metapath APA is shown below:

![APA](https://i.imgur.com/TnPyamV.png)

As word2vec does not generate embeddings for unseen words, test applications in our case, we trained a decision tree regressor using the true embeddings for training application as the labels, and the average of all the embeddings of each application’s associated api as the training data. Using this regressor we are able to generate embeddings for test applications using its associated api appear in the training corpus.

### Node2Vec

In node2vec, the entire Heterogeneous Information Network is regarded as an large homogeneous graph and the only theoretical difference to the word2vec approach is the random walk procedure. The graph traversal method is based on a graph where all different types of edges are merged together to be one. This change is adapting to the inability of node2vec to traverse according to a metapath but instead a truly random walk with no specific rules restricting where the next node would be. We choose a return parameter of 2 and a in-out parameter of 1 empirically to perform walks beginning on each app node for 100 times. This results in a corpus similar to the word2vec approach, so we could use the same methodology to match and predict different distributions of app and API embeddings.

As node2vec also does not generate embeddings for test appication, we use the similar approach to generate the embeddings for test application using associated API embeddings.

![node2vec](https://i.imgur.com/auK5rqj.png)

### Metapath2Vec

![metapath2vec equation](https://i.imgur.com/AINz4lr.png) (1)

Metapath2Vec is used as a technique of sampling our next node. We sample our next node using equation (1). Let's use an example to illustrate the process.

Imagine that we have these matrices set up, and our defined metapath is **ABA**. Our metapath-chosen sentence will look like "app_A API_Y API_Z app_B". An sample matrix looks something like the following:  
![Matrices](https://i.imgur.com/XIYFrc3.png)

Simplified steps:

1. Pick an app. This will replace app_A.
2. Go to the matrix corresponding to the metapath. For example, the first path in **ABA** is A, so we will look at the A matrix.
3. Go to the row corresponding to the app or API that was chosen.
4. Pick an API. Within a row, the APIs that have a value of 1 is picked using a uniform probability.
5. Repeat 2, 3, and 4 until you are ready to pick an app (app_B). With the API that was chosen (API_Z), look at the column and pick an app that has value 1 with uniform probaility.

![ABA](https://i.imgur.com/8N8IeYi.png)
![ABPBPBBPA](https://i.imgur.com/Wi5C3KW.png)
![ABABBABBBABBBBABBBBBA](https://i.imgur.com/etgIVjM.png)


## Results

Let's take a look at the different accuracies for the original HinDroid approach and the HinDroid approach with additional embedding techniques.

**HinDroid:**

| Metapath | train_acc | test_acc | F1     | TP    | FP   | TN    | FN   |
|----------|-----------|----------|--------|-------|------|-------|------|
| AA       | 1.0000    | 0.9561   | 0.9562 | 158   | 10   | 147   | 4    |
| APA      | 1.0000    | 0.9373   | 0.9412 | 155   | 14   | 145   | 6    |
| ABA      | 0.9149    | 0.8558   | 0.8671 | 147   | 27   | 130   | 19   |
| APBPA    | 0.9040    | 0.8339   | 0.8408 | 140   | 32   | 126   | 22   |

**Reduced:**

| Metapath | train_acc | test_acc | F1     | TP    | FP   | TN    | FN   |
|----------|-----------|----------|--------|-------|------|-------|------|
| AA       | 1.0000    | 0.9561   | 0.9562 | 158   | 10   | 147   | 4    |
| APA      | 1.0000    | 0.9373   | 0.9412 | 155   | 14   | 145   | 6    |
| ABA      | 0.9149    | 0.8558   | 0.8671 | 147   | 27   | 130   | 19   |
| APBPA    | 0.9040    | 0.8339   | 0.8408 | 140   | 32   | 126   | 22   |

**Word2Vec**

| Metapath  | Accuracy | F1     | TN  | FP | FN | TP  |
|-----------|----------|--------|-----|----|----|-----|
| ABA       | 95.06%   | 95.02% | 639 | 32 | 34 | 630 |
| ABPBA     | 94.61%   | 94.59% | 634 | 37 | 35 | 629 |
| APA       | 95.73%   | 95.68% | 647 | 24 | 33 | 631 |
| APBPA     | 94.61%   | 94.55% | 638 | 33 | 39 | 625 |

**Node2Vec**

| Metapath  | Accuracy | F1     | TN  | FP | FN | TP  |
|-----------|----------|--------|-----|----|----|-----|
| N/A       | 97.75%   | 97.77% | 648 | 18 | 12 | 657 |

**Metapath2Vec:**

| Metapath              | train_acc | test_acc | F1     | TP    | FP   | TN    | FN   |
|-----------------------|-----------|----------|--------|-------|------|-------|------|
| AA                    | 0.9736    | 0.9476   | 0.9466 | 621   | 27   | 644   | 43   |
| APA                   | 0.9955    | 0.9296   | 0.9277 | 603   | 33   | 638   | 61   |
| ABA                   | 0.9864    | 0.9633   | 0.9626 | 630   | 15   | 656   | 34   |
| APBPA                 | 0.9900    | 0.9438   | 0.9419 | 608   | 19   | 652   | 56   |
| ABPBA                 | 0.9982    | 0.9524   | 0.9524 | 614   | 10   | 661   | 50   |
| ABPBPBBPA             | 0.9973    | 0.9476   | 0.9455 | 607   | 13   | 658   | 57   |
| ABABBABBBABBBBABBBBBA | 0.9545    | 0.9026   | 0.9027 | 603   | 69   | 602   | 61   |

## Conclusion

From our initial testing, all of our proposed graph embedding techniques are able to achieve similar accuracy and metrics scores. Although it does not seem that the different graph embeddings obtained a higher accuracy score, we believe that using these other graph embedding techniques not only matches with the results of HinDroid, but are also more robust in the sense that hackers are not able to easily rearrange APIs to avoid detection.  

There is definitely further research to do. One being where dummy nodes are added. This will provide us with more evidence of the robustness of the different graph techniques used. The clusters that are present in the visualizations are suggestive of further analysis. Also, we should test out several more different meta-paths. We can test which meta-paths work the best and investigate why it works. We could also do concatenation of embeddings from different meta-paths to form a longer representation and test its effectiveness. 

## References

[1] Passi, Harpreet. Introduction to Malware: Definition, Attacks, Types and Analysis. GreyCampus  
[2] Hou, Shifu and Ye, Yanfang and Song, Yangqiu and Abdulhayoglu, Melih. 2017. HinDroid: An Intelligent Android Malware Detection System Based on Structured Heterogeneous Information Network.  
[3] Mikolov, Tomas and Corrado, Greg and Chen, Kai and Dean, Jeffrey. 2013. Efficient Estimation of Word Representations in Vector Space.  
[4] Grover, Aditya and Leskovec, Jure. 2016. node2vec: Scalable Feature Learning for Networks.  
[5] Dong, Yuxiao and Chawla, Nitesh and Swami, Ananthram. 2017. metapath2vec: Scalable Representation Learning for Heterogeneous Networks  
