from ultralytics import YOLO

model = YOLO('yolov5n.yaml').load('Submission/best.pt')

model.predict('Submission/test_images', save=True, imgsz=320, conf=0.5)

