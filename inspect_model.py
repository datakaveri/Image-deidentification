from ultralytics import YOLO
import torch
import os

weights = 'license_plate_detector.pt'
print('exists', os.path.exists(weights))
model = YOLO(weights)
print('ckpt_path', getattr(model, 'ckpt_path', None))
print('names', model.names)
print('n_classes', len(model.names))
print('cuda_available', torch.cuda.is_available(), torch.cuda.device_count())

imgs = sorted([f for f in os.listdir('train/images') if f.lower().endswith(('.jpg', '.png'))])[:1]
print('sample images', imgs)
if imgs:
    res = model.predict(source=os.path.join('train/images', imgs[0]), conf=0.25, imgsz=640, device='cpu', verbose=False)
    if res:
        r = res[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is not None:
            print('boxes', len(boxes))
            if hasattr(boxes, 'cls'):
                print('classes', boxes.cls.cpu().numpy())
            if hasattr(boxes, 'xyxy'):
                print('bboxes', boxes.xyxy.cpu().numpy())
