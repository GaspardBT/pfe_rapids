# PFE Rapids
End of Studies projects at CentraleSupelec. The goal is to discover and test the Rapids ecosystem, focusing on the librairy cuML.
## Installation
Follow the following steps:
 - Install Conda
 - Create working environment for Rapids using the following explainer: [link](https://rapids.ai/start.html). It will install all the require depencie to run cuml.
 - Activate the environment `conda activate env_name`.
 - Install the others depencies the will be listed in for each sub-project.


## Data 
You can download the data used using the following links::
 - [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data)
 - [Detecting Malicious URLs](http://www.sysnet.ucsd.edu/projects/url/) using the svm light version.
## Hello Scripts

### [Clustering Benchmark](./hello_scripts/bench_cluster.py)
   
   Script to benchmark different clustering algorithms. Code adapted from [here](https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html).
  
### [kNN](./hello_scripts/test_knn.py) 
Test cuml kNN on real data. Code adapted from [here](https://www.kaggle.com/cdeotte/rapids-gpu-knn-mnist-0-97/notebook)
### [Kmeans 101](./hello_scripts/hello_kmeans.py) 
A script showing to most basic use of cuML Kmeans implementation.
## [Image App](./img_app/) 

Test of a integration of cuML in a Flask API.  
Run the app with: `python app.py`  
You can pretrained the model with the `model_maker.py` script, make sure to properly set the dataset path.
## Urls Classification

Set the rigth dataset path and launch the script using `python trainer_standalone.py`
## Full Stream
 - Set the right dataset path in this [script](full_stream/src/mock_producer/main.py).
 - Be sure to have Kafka running you can follow [this](full_stream/command.md)
 - Launch the mock producer `python src/mock_producer/main.py` 
 - Launch the trainer `python src/trainer/main.py`
 - Launch the metric collector `python src/metrics_garbage/main.py`


## Metrics Analysis
Notebooks to plot metrics analysis.