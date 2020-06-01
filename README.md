# HinDroid-with-Embeddings

![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/davidzz/hindroid-xl)

## Overview

Malware detection for android applications is an expanding field with the introduction of the [Hindroid](https://www.cse.ust.hk/~yqsong/papers/2017-KDD-HINDROID.pdf) model (DOI:[10.1145/3097983.3098026](https://doi.org/10.1145/3097983.3098026)). It proposes a method that transforms the semantic relationships between Android applications and their decompiled source code to a Heterogeneous Information Network (HIN) and uses similarities from various meta-paths between apps to construct a model for malware classification. To further explore the field, we aim to extend the HinDroid model to improve the accuracy in specific subsets of the AMD dataset. Our effort will be focused on finding better representations for both apps as well as APIs and discovering methods to incorporate them as additional features in a new model. In the meantime, we plan to evaluate how the proposed model captures the features that are relevant to the classification task and compare to that of the HinDroid baseline. Our contributions can be utilized in systems where the analysis of malware and interpretable features are more important than mere detection.

## Usage
Docker image on Docker Hub:
[davidzyx/hindroid-xl](https://hub.docker.com/repository/docker/davidzz/hindroid)

On Datahub, create a pod using a custom image with 4 CPU and 32 GB of memory. If you use another configuration, use at least 6GB memory per CPU.
```bash
launch.sh -i davidzz/hindroid-xl -c 4 -m 24
```

Modify the config file located in `config/data-params.json`. If you want to run a test drive, use `config/test-params.json`. Put either `data` or `test` as the first argument.

The HinDroid baseline uses the driver file `run.py` with 3 targets: `ingest`, `process`, and `model`. Put each target space-separated as arguments in the call. To run the whole pipeline, use
```bash
python run.py data ingest process model
```

`process` target will save `.npz` files in `data/processed/` for generating various embeddings.

## System Prerequisites and Definitions

- APK - Executable file for Android
![executables](https://i.imgur.com/RrGNIoG.png)
- Smali code - Human readable code decompiled from Dalvik bytecode contained the APK
- Heterogeneous Information Network (HIN) - A graph where its nodes may not be of the same type
- API Extraction
  - Use regex to match specific patterns
  - API calls and method blocks
![api](https://i.imgur.com/nx3rKKv.png)

## HinDroid

Our project is an improvement and exploration of different embedding techniques based on a previous paper - HinDroid. Therefore, some techniques and basic data structures are based on what HinDroid outlined, such as how data was extracted and structured.

Data was extracted from APKs downloaded from APKPure.com, and the smali code was extracted by using APKTool to reverse engineer the APKs.

3 matrices were borrowed from HinDroid: A, B, and P matrix. Below you will see a description for each matrix and an example.

**A** matrix tells us information about whether APIs are within the same app. Each API within an app will have 1, else it will have 0.

![A matrix](https://i.imgur.com/fGIEH9c.png)

**B** matrix tells us information about whether APIs are in the same code block. If API_1 and API_2 are in the same code block then the corresponding spot in the matrix will have 1, else it will have 0.

![B matrix](https://i.imgur.com/5MJL9ff.png)

**C** matrix tells us information about whether APIs are in the same package. If API_1 and API_2 are in the same package then the corresponding spot in the matrix will have 1, else it will have 0.

![C matrix](https://i.imgur.com/X9n40VE.png)

## Embedding Techniques

### Word2Vec

Word2Vec is a powerful approach to generate embeddings for words using its context in a corpus. Decades of work and research using this idea has led to state of the art models powering the technologies we are using everyday such as Siri, Google Translate, next word prediction on our smartphone keyboard and etc. If you are interested of knowing more, [this](http://jalammar.github.io/illustrated-word2vec/) great blog post provide detailed explanation with excellent graphical example. 

As Word2Vec cannot be directly applied to application source code and matrices from HinDroid, our data ingestion pipeline generates text corpus by traversing the graphs following an user defined metapaths and the length of a random walk. Using a metapath `ABPBA` with random walk length 5000, the text corpus may look like 

`app_3 -> api_500 -> api_321 -> api_234 -> api_578 -> app_321 -> api_123…`

where `app_3` and `api_500` are connected by matrix A, api_500 and api_321 are connected by matrix B and so on. During each random walk, the metapath will be repeated until the length of the walk is met by randomly selecting an application or api node that is directly connected to the starting node. In the text corpus, an application node will always be followed by an api node, where an api node will be followed by an application node only if the next matrix in the metapath is matrix A. 
 
In a graph where there are two types of nodes: application and api, Word2Vec is the first approach that we attempt to capture the relationship beyond application and apis that have a direct connection in the graph. This traditional and powerful NLP embeddings techniques helps us to learn the similarity between applications not just limited to the shared api, and also the ability to identify the clusters connection between application and api that do not always have a direct connection. Using gensim’s word2vec model, we are able to generate vector embeddings for each application and api found in the text corpus. We successfully converted decompiled Android source code into a vector of numbers for each application and this information can be easily used in a machine learning model.
 
To evaluate the effectiveness of the generated embeddings, we visualize the embedding clusters by applying dimensionality reduction into two dimensional vectors. The embedding visualization for metapath ABA, APA, ABPBA and APBPA are shown below:

![ABA](https://i.imgur.com/a6DBPmO.png)

![ABPBA](https://i.imgur.com/4lW3npg.png)

![APA](https://i.imgur.com/TnPyamV.png)

![APBPA](https://i.imgur.com/NZq0zxO.png)

The graph shows promising results and the clusters are separated nicely between benign and malware applications. As word2vec does not generate embeddings for unseen words, test applications in our case, we trained a decision tree regressor using the true embeddings for training application as the labels, and the average of the embeddings of each application’s associated api as the training data. Using this regressor we are able to generate embeddings for test applications using its associated api appear in the training corpus.

### Node2Vec

### Metapath2Vec

![metapath2vec equation](https://i.imgur.com/AINz4lr.png) (1)

Metapath2Vec is used as a technique of sampling our next node. We sample our next node using equation (1). Let's use an example to illustrate the process.

Imagine that we have these matrices set up, and our defined metapath is **ABA**. Our metapath-chosen sentence will look like "app_A API_Y API_Z app_B".
![Matrices](https://i.imgur.com/XIYFrc3.png)

Steps:

1. We choose an app. In this case there are only two apps. Suppose we choose app_0.  
![app_0](https://i.imgur.com/xWePMRv.png)
2. Our first path in out metapath is **A**. So, now we will look in the **A** matrix and the row for app_0. We see that app_0 contains API_1 and API_2. Therefore we will sample API_1 and API_2 with a uniform probability, where each API has a probability of 0.5 of getting chosen. Let's suppose we choose API_2.  
![api-1](https://i.imgur.com/uFUIrQ3.png)
3. Now out path moves on to **B**. We go to the **B** matrix and look at the row for API_2. We see that we can either choose API_1 or API_2. They both will have a probability of 0.5 of getting chosen. Let's suppose we choose API_1.  
![api-2](https://i.imgur.com/lwWLAg8.png)
4. Our path moves to the last spot in the metapath, which is **A**. We go back to our A matrix. Look at the column for API_1, and we see that we are able to choose either app_0 or app_1. Suppose we choose app_1. Our resulting sentence would look like the following.  
![final path](https://i.imgur.com/iGzXhfW.png)

## Results

Let's take a look at the different accuracies for the original HinDroid approach and the HinDroid approach with additional embedding techniques.

HinDroid:

| metapath | train_acc | test_acc | F1     | TP    | FP   | TN    | FN   |
|----------|-----------|----------|--------|-------|------|-------|------|
| AA       | 1.0000    | 0.9561   | 0.9562 | 158   | 10   | 147   | 4    |
| APA      | 1.0000    | 0.9373   | 0.9412 | 155   | 14   | 145   | 6    |
| ABA      | 0.9149    | 0.8558   | 0.8671 | 147   | 27   | 130   | 19   |
| APBPA    | 0.9040    | 0.8339   | 0.8408 | 140   | 32   | 126   | 22   |

Reduced:

| metapath | train_acc | test_acc | F1     | TP    | FP   | TN    | FN   |
|----------|-----------|----------|--------|-------|------|-------|------|
| AA       | 1.0000    | 0.9561   | 0.9562 | 158   | 10   | 147   | 4    |
| APA      | 1.0000    | 0.9373   | 0.9412 | 155   | 14   | 145   | 6    |
| ABA      | 0.9149    | 0.8558   | 0.8671 | 147   | 27   | 130   | 19   |
| APBPA    | 0.9040    | 0.8339   | 0.8408 | 140   | 32   | 126   | 22   |

Metapath2Vec:

| metapath              | train_acc | test_acc | F1     | TP    | FP   | TN    | FN   |
|-----------------------|-----------|----------|--------|-------|------|-------|------|
| AA                    | 0.9736    | 0.9476   | 0.9466 | 621   | 27   | 644   | 43   |
| APA                   | 0.9955    | 0.9296   | 0.9277 | 603   | 33   | 638   | 61   |
| ABA                   | 0.9864    | 0.9633   | 0.9626 | 630   | 15   | 656   | 34   |
| APBPA                 | 0.9900    | 0.9438   | 0.9419 | 608   | 19   | 652   | 56   |
| ABPBA                 | 0.9982    | 0.9524   | 0.9524 | 614   | 10   | 661   | 50   |
| ABPBPBBPA             | 0.9973    | 0.9476   | 0.9455 | 607   | 13   | 658   | 57   |
| ABABBABBBABBBBABBBBBA | 0.9545    | 0.9026   | 0.9027 | 603   | 69   | 602   | 61   |

### Class Labels Separation

![Labels](https://i.imgur.com/cdFOD6m.jpg)

## Conclusion

From our initial testing, all of our proposed graph embedding techniques are able to achieve similar accuracy and metrics scores. Although it does not seem that the different graph embeddings obtained a higher accuracy score, we believe that using these other graph embedding techniques not only matches with the results of HinDroid, but are also more robust in the sense that hackers are not able easily rearrange APIs to avoid detection.  

There is definitely further research to do. One being where dummy nodes are added. This will provide us with more evidence of the robustness of the different graph techniques used. Also, we should test out several more different metap-paths. We can test which meta-paths work the best and why it works.

## References

[1] Passi, Harpreet. Introduction to Malware: Definition, Attacks, Types and Analysis. GreyCampus  
[2] Hou, Shifu and Ye, Yanfang and Song, Yangqiu and Abdulhayoglu, Melih. 2017. HinDroid: An Intelligent Android Malware Detection System Based on Structured Heterogeneous Information Network.  
[3] Mikolov, Tomas and Corrado, Greg and Chen, Kai and Dean, Jeffrey. 2013. Efficient Estimation of Word Representations in Vector Space.  
[4] Grover, Aditya and Leskovec, Jure. 2016. node2vec: Scalable Feature Learning for Networks.  
[5] Dong, Yuxiao and Chawla, Nitesh and Swami, Ananthram. 2017. metapath2vec: Scalable Representation Learning for Heterogeneous Networks  