# GanforKGE

Data


In 'data' directory, we provide some examples of WN18RR and WN11 dataset. 


WN18RR is a dataset for link prediction, each line is a triple, respectively representing head, relation and tail


WN11 is a dataset for triple classification, each line contains a triple and a label, 1 for positive and -1 for negative


Run


link prediction task


python trainGan.py --embeddingname TransH --gname Translation --dname ConvE


which means TransH model is served as the role of generator and convolutional neural network is served as the role of discriminator.


Other optimal settings are listed as follows,


embeddingname: TransE, TranH, TransD


gname: Translation, FC, ConvE, ConvTransE


dname: Translation, FC, Conv


triple classification task


python trainTripleClassification.py --embeddingname TransH --gname Translation --dname ConvE

