from ultralytics import YOLO

model = YOLO('yoloSkipSmall.yaml')

model.train(data='mycoco.yaml', epochs=50, device="0,1",
            save=True, save_period=5)
