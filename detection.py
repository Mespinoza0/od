import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

def main():
    with open('data1.yaml', mode='r') as f:
        data_yaml = yaml.load(f, Loader=SafeLoader)

    labels = data_yaml['names']
    
    yolo = cv2.dnn.readNetFromONNX('./Model_3/weights/best.onnx')
    if yolo.empty():
        print("Error: Failed to load YOLO model.")
        return
    else:
        print("YOLO model loaded successfully.")
    
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Change this line to capture the stream from the Raspberry Pi camera feed
    cap = cv2.VideoCapture("http://11.21.40.241:8080/?action=stream")
    
    if not cap.isOpened():
        print("Error: Could not access the camera stream.")
        return

    print("Press 'q' to quit the application at any time.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            else:
                print("Captured frame successfully")

            image = frame.copy()
            row, col, d = image.shape

            max_rc = max(row, col)
            input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
            input_image[0:row, 0:col] = image

            INPUT_WH_YOLO = 640
            blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
            yolo.setInput(blob)

            start_time = cv2.getTickCount()
            preds = yolo.forward()
            end_time = cv2.getTickCount()
            time_taken = (end_time - start_time) / cv2.getTickFrequency()
            print(f"Frame processed in: {time_taken:.3f} seconds")

            detections = preds[0]
            boxes = []
            confidences = []
            classes = []

            image_w, image_h = input_image.shape[:2]
            x_factor = image_w / INPUT_WH_YOLO
            y_factor = image_h / INPUT_WH_YOLO

            for i in range(len(detections)):
                row = detections[i]
                confidence = row[4]
                if confidence > 0.6:
                    class_score = row[5:].max()
                    class_id = row[5:].argmax()

                    if class_score > 0.25:
                        cx, cy, w, h = row[:4]
                        left = int((cx - 0.5 * w) * x_factor)
                        top = int((cy - 0.5 * h) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)

                        box = np.array([left, top, width, height])

                        confidences.append(confidence)
                        boxes.append(box)
                        classes.append(class_id)

            boxes_np = np.array(boxes).tolist()
            confidences_np = np.array(confidences).tolist()

            index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.7, 0.3)  # Lower NMS threshold

            if index is None or len(index) == 0:
                print("No objects detected.")
                continue

            index = index.flatten()

            object_detected = False

            for ind in index:
                x, y, w, h = boxes_np[ind]
                bb_conf = int(confidences_np[ind] * 100)
                class_id = classes[ind]
                class_name = labels[class_id]

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
                cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)
                text = f'{class_name}: {bb_conf}%'
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)  

                object_detected = True

            if object_detected:
                print("OBJECT(S) DETECTED")
            else:
                print("OBJECT NOT DETECTED")

            cv2.imshow('YOLO Object Detection', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break

    except KeyboardInterrupt:
        print("Exiting gracefully...")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
