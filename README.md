# Data Efficient Language-Supervised Zero-Shot Recognition with Optimal Transport Distillation
This code provides a PyTorch implementation for **OTTER** (**O**ptimal 
**T**ranspor**t** distillation for **E**fficient zero-shot **R**ecognition), 
as described in the paper [ Data Efficient Language-Supervised Zero-Shot 
Recognition with Optimal Transport Distillation]().

## Installation
First, git clone the repository
```
git clone https://github.com/facebookresearch/OTTER.git
```
Then, install required packkages using pip
```
conda create --name otter --python=python3.6
pip install -r requirements.txt
```

## Data preparation
Download the conceptual caption dataset for training and google open images's test set for evaluation. Assume that the two datasets are placed under 
```
DATA_ROOT/
  cc/
    training/
      ... (images)
  open-images/
    test/
      ... (images)
```
Note that not all images in the conceptual caption datasets are available. In our case, we downloaded 2911810 images from conceptual caption. 

## Single node training 
You can launch training on a single node with the following command
```
. launch.sh
```
