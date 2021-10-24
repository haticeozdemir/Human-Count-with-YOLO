# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 21:50:28 2021

@author: HaticeOzdemir
"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "C:/Users/HaticeOzdemir/Downloads/yolo.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "C:/Users/HaticeOzdemir/Downloads/AIComputerVision-master/img/people3.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), minimum_percentage_probability=30)
class1=[]
for eachObject in detections:
    

    if eachObject["name"] != "person":
        continue
    else:
        
        sonuc =eachObject["name"]
        class1.append(sonuc)
print(class1)
print(len(class1))

#print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        
    
    
  
    