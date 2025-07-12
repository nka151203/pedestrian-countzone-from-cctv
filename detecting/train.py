from ultralytics import YOLO
model = YOLO('./model/yolo11m.pt')

epoch_num = 40
batch_size = -1
img_size = 640
project_name = "./model"  # result saved folder
yolo_path = "./detecting/yolo_train.yaml"
name = "yolov11m_mot20_det"

results = model.train(
    data=yolo_path,
    epochs=epoch_num,
    batch=batch_size,
    imgsz=img_size,
    project=project_name,
    name=name,
    save=True,          # save model on
    save_period=10     # save model after 10 epochs
)