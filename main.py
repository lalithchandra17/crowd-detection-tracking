import cv2
import csv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

sources = ["video1.mp4", "video2.mp4"]
current_source = 0

cap = cv2.VideoCapture(sources[current_source])

tracker_config = "botsort.yaml"

out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      20,
                      (640, 480))

csv_file = open("tracking_data.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "ID", "X1", "Y1", "X2", "Y2", "Camera"])

frame_id = 0

zone_x1, zone_y1, zone_x2, zone_y2 = 100, 100, 400, 400

while True:
    ret, frame = cap.read()
    if not ret:
        cap.release()
        current_source = (current_source + 1) % len(sources)
        cap = cv2.VideoCapture(sources[current_source])
        continue

    frame_id += 1

    results = model.track(frame, persist=True, tracker=tracker_config)

    annotated_frame = frame.copy()

    cv2.rectangle(annotated_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255,0,0), 2)

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                id = int(box.id) if box.id is not None else 0

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                color = (0,255,0)

                if zone_x1 < cx < zone_x2 and zone_y1 < cy < zone_y2:
                    color = (0,0,255)
                    cv2.putText(annotated_frame, "ALERT", (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated_frame, f"ID: {id}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                csv_writer.writerow([frame_id, id, x1, y1, x2, y2, current_source])

    if results[0].boxes is not None:
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"People Count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(annotated_frame, f"Camera: {current_source}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Smart Surveillance System", annotated_frame)

    out.write(annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        cap.release()
        current_source = (current_source + 1) % len(sources)
        cap = cv2.VideoCapture(sources[current_source])

    if key == 27:
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
