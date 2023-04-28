from ultralytics import YOLO

model = YOLO('yolov5n.yaml')

model.train(data='mycoco.yaml', epochs=100,
            save=True, save_period=5)

