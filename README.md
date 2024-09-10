# masters-project
This project refers to the Master's project in University of Manchester for ___Multi-modal Deep Learning for Alzheimer’s Disease Prediction__

The files includes:  

  [tabular-preprocess.py](https://github.com/ZYX-yixinzhang/masters-project/blob/main/tabular-preprocess.py): read the tabular data and preprocess  
  
  [image-preprocess.py](https://github.com/ZYX-yixinzhang/masters-project/blob/main/image-preprocess.py): read the image data and preprocess  
  
  [models/](https://github.com/ZYX-yixinzhang/masters-project/tree/main/models): all the models includes in the project  

  [explaination.py](https://github.com/ZYX-yixinzhang/masters-project/tree/main/explanation.py): explanation of model  
  
  [visualisation.py](https://github.com/ZYX-yixinzhang/masters-project/blob/main/visualisation.py): visulisation of training
  
It can be run on:
  - python 3.8+
  - pytorch 2.1.0+cu118
  - pandas 1.4.3
  - ants.py 0.2.3
  - scikit learn 1.1.1
  - shap 0.46.0
  - matplotlib 3.8.1
  - numpy 1.23.5
  
## Dataset
This project used [the Alzheimer's Disease Neuralimaging Initiatives (ADNI) dataset](https://adni.loni.usc.edu/).  

According to the [ADNI dataset use agreement](https://adni.loni.usc.edu/revised-adni-data-use-agreement/), the raw data files are __not__ uploaded  

## How to run this project
The project can be run with (you should have the ADNI dataset in ```./dataset/dataset.csv```:
  ```
  python3 tabular-preprocess.py
  ```
it will generate two ```.csv``` files: train.csv and test.csv in the main directory  

Then run (you should have the images in ```./dataset/scans/ADNI```):
```
  python3 image-preprocess.py
```
it will generate a folder in the main directory, which includes all the preprocessed images  

After that, you can run the models you want:
```
  python3 models/cnngru.py
```

and explain it with:
```
  python3 explanation.py
```
on the model you want  

or visualise the training process with:
```
  python3 visualise.py
```
