from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
    # model = YOLO('yolov8s-pose.yaml')  # build a new model from YAML
    # model = YOLO('yolov8m-pose.yaml')  # build a new model from YAML
    # model = YOLO('yolov8m-pose.yaml')  # build a new model from YAML
    model.train(data='yak-pose.yaml', epochs=100, imgsz=640)

