# YOLO modeli (You-Look-Only-Once) Olasılık Hesabı Var
# Semantic Segmentation (No objects, just pixels)
# Classifacation + LOcalization (Single Object)
# Object Detection (Multiple Object)
# Instance Segmentation (Multiple Object)

# Non-max Supression (Güven Algoritması)

'''
pip install ultralytics
YOLO modellerini kullanmamızı sağlıyor
'''

import cv2
from ultralytics import YOLO

video = 'trafik-1.mp4'

cap = cv2.VideoCapture(video)

yolo_model = YOLO('yolov8n.pt')

while cap.isOpened():
    # Açık mı değil mi kontrol
    
    read, frame = cap.read() # hasFrame
    
    if read:
        
        results = yolo_model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Trafik Nesne Tespiti', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
        
    else:
        break
        
cap.release()
cv2.destroyAllWindows()
        
    
    