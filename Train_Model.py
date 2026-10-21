from ultralytics import YOLO

# Memuat model dasar YOLOv8 (versi nano 'yolov8n.pt' agar proses training lebih cepat dan ringan)
model = YOLO('yolov8n.pt')

# Memulai proses training menggunakan data.yaml dari folder unduhan
model.train(data='Pothole-1/data.yaml', epochs=25, imgsz=512, batch=16)