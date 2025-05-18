import torch
import cv2
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Vehicle classes
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']

# Start webcam
cap = cv2.VideoCapture(0)

# Timer related variables
signal_timer = 0
last_update_time = time.time()

def calculate_timer(vehicle_count):
    if vehicle_count <= 5:
        return 30
    elif vehicle_count <= 10:
        return 60
    else:
        return 90

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)
    df = results.pandas().xyxy[0]
    vehicle_count = sum(df['name'].isin(vehicle_classes))

    # Render
    annotated_frame = results.render()[0].copy()

    # Update timer every 10 seconds
    current_time = time.time()
    if current_time - last_update_time > 10:
        signal_timer = calculate_timer(vehicle_count)
        last_update_time = current_time

    # Show vehicle count and signal timer
    cv2.putText(annotated_frame, f'Vehicle Count: {vehicle_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(annotated_frame, f'Signal Timer: {signal_timer}s', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display
    cv2.imshow("Smart Traffic Signal System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
