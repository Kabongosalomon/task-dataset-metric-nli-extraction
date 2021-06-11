# task-dataset-metric-nli-extraction

This program produces the test data for classification over a set of predefined task#dataset#metrics#software labels.
Given input a pdf file, it scrapes the text from the file using the Grobid parser, subsequently generating the test data file for input to the neural network classifier.


## Acknowledgement: 
This program reuses code modules from IBM's science-result-extractor (https://github.com/IBM/science-result-extractor). A reference url to their paper on the ACL anthology is https://www.aclweb.org/anthology/P19-1513