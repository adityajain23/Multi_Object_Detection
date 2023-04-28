from ultralytics import YOLO

model = YOLO('yoloAnchors.yaml')

model.train(data='mycoco.yaml', epochs=50,
            device="0", save=True, save_period=5)
