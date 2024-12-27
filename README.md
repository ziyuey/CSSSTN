English | [简体中文](README_CN.md)

## Code Description
Source code for the paper:  

In this repository, we provide code for data preprocessing, classifier training, and generator training. You can easily use this code on your own dataset.

## How to Run the CSSSTN Code
1. Perform wavelet transform preprocessing on the EEG data.
2. Pre-train a CNN classification model and save it.
3. Load the pre-trained model into `CSSSTN.py` and run CSSSTN.

## File Description
- model_save folder: Used to save the trained models.
- CSSSTN.py: Contains the code for training the generator model. Lines 132 and 133 are used to load the target and source classification models, respectively.
- CNN.py: Contains the code for training the classification model. Line 129 is used to save the trained model.
- G_D.py: Includes the generator and CNN parts of the CSSSTN model.
