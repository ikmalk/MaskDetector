# MaskDetector

Soft Computing projects

## Requirements
```
cv2
numpy
tensorflow
pickle
matplotlib
```

## How to run

### Data preparation

1. Create a new folder for datasets
2. Create folders inside datasets folder with labels as folder name E.g
```
datasets
  |-- Mask (Label 1)
  |-- NoMask (Label 2)
```
3. Download datasets from internet, my dataset is from https://github.com/cabani/MaskedFace-Net
4. Put images in approprirate file in datasets folder
5. Modify path variable and categories in DatasetsGenerator.py
6. Run DatasetsGenerator.py

### Training Model

1. Modify and make sure the CATEGORIES variable matches the folder name in datasets
2. If DatasetsGenerator.py runs without error, there should be a pickle file in directory
3. Run trainer.py

### Testing

1. Modify and make sure the CATEGORIES variable matches the folder name in datasets
2. Run test.py






