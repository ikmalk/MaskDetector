# MaskDetector

Soft Computing projects

## Requirements
```
cv2
numpy
tensorflow
pickle
plotly
```

## How to run

### Data preparation

1. Create a new folder in directory for named "datasets"
2. Create folders inside datasets folder with labels as folder name E.g
```
datasets
  |-- Mask (Label 1)
  |-- NoMask (Label 2)
```
3. Download datasets from internet, my dataset is from https://github.com/cabani/MaskedFace-Net
4. Put images in approprirate file in datasets folder
5. Run DatasetsGenerator.py

### Training Model

You can skip this process by downloading the entire model folder [here](https://drive.google.com/drive/folders/1Z3sGnD-NP3jrR3Tqjx6XQMdyaaFC20GI?usp=sharing) and put it inside the code folder. (Can't put in github because the model is over 100mb)

1. If DatasetsGenerator.py runs without error, there should be a pickle file in directory
2. Run trainer.py

### Getting accuracy

1. Create a new folder in directory named "testdata" with same folder structure as datasets folder
2. Put images that is not in the datasets folder inside the relevant folder in the testdata folder
3. Run stats.py

### Implementaion
1. Make sure webcam is enabled
2. Run MaskDetector.py

Example of MaskDetector out put

![1](https://github.com/ikmalk/MaskDetector/blob/main/NoMask.JPG)
![2](https://github.com/ikmalk/MaskDetector/blob/main/Mouth.JPG)
![3](https://github.com/ikmalk/MaskDetector/blob/main/Mask.JPG)
![4](https://github.com/ikmalk/MaskDetector/blob/main/Nose.JPG)



