import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

yolo_s_model = YOLO('yolov8n-seg.pt')
names = yolo_s_model.names # Class ları alıyoruz

cap = cv2.VideoCapture('trafik-2.mp4')
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter('trafik_segmentation.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w,h))

while True:
    read, frame = cap.read()
    
    if not read:
        print('Görüntü bitti ya da yok...')
        break
    
    results = yolo_s_model.predict(frame)
    
    annotator = Annotator(frame, line_width=2 # Çizgi kalınlığı
                          )
    
    if results[0].masks is not None: # segmantasyon maskeleri (pikseller) tespit edildiyse
        
        clss = results[0].boxes.cls.cpu().tolist() # Sınıf etiketleriyle piksellerin eşlenmiş halinin liste hali
        masks = results[0].masks.xy # Piksellerin konumlarını alıyor
        
        for mask, cls in zip(masks, clss): # Koumlarla isimleri eşleştiriyoruz
        
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(int(cls), True),
                               label=names[int(cls)])
            
    out.write(frame)
    cv2.imshow('YOLO Trafik Segmantasyonu', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print('Çıkış yapılıyor')
        break
    
out.release()
cap.release()
cv2.destroyAllWindows()

            
        