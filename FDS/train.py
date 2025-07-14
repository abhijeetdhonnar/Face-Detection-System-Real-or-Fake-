from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')  # Choose yolov8n.pt, yolov8s.pt, etc.
    model.train(data='Dataset/SplitData/data.yaml', epochs=3)

if __name__ == '__main__':
    main()
