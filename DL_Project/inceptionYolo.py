from ultralytics import YOLO

model = YOLO('yoloInception.yaml')

model.train(data='mycoco.yaml', epochs=50,
            save=True, save_period=5)
