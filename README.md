# Diabetic-Retinopathy-Detection
This project was a collaboration between me and my colleagues for our AI project in University, the purpose of the project is to detect 
Diabetic Retinopathy(DR) early so people can receive treatment before it becomes incurable
# Description
- The models used can classify images from 0 to 4, 0 meaning doesn't have DR, while 4 being Proliferative DR
- we used 4 pre-trained models which are DenseNet121, ResNet50, VGG16, EfficientNetB5
- DenseNet yielded the best accuracy
- we also made a prototype GUI to showcase the project
# Dataset and Pre-Processing
- APTOS 2019 Dataset was used to train these models 
- In addition we did some image augmentation with python scripts on classes 3 and 4 because their data were not as much as the other classes
- we applied gray scale, circle crop and gaussian blur on the images so the features of the images can be clearer and easier for the models to work on

**Image Before pre-processing**
![unprocessed image](/sample images/unprocessed.png)

**Image After pre-processing**
![pre-processed image](/sample images/processed.png)

# GUI Prototype
![GUI image](/sample images/test result.png)

