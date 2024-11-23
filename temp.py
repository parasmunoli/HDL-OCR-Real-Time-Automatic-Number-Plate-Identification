import cv2
import inference
import supervision as sv
import pytesseract
import threading

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global variables
frame_skip_count = 0

# Lock for synchronization in threading
lock = threading.Lock()

annotator = sv.BoxAnnotator()

def render(predictions, image):
    global frame_skip_count
    with lock:
        frame_skip_count += 1
        if frame_skip_count % 2 != 0:  # Process every 2nd frame
            return

    if isinstance(predictions, list):
        return

    detected = []
    data = predictions['predictions']
    for prediction in data:
        if prediction.get('class') == 'Number Plate' or prediction.get('class_id') == 0:
            x, y, w, h = map(int, [prediction.get('x', 0), prediction.get('y', 0), prediction.get('width', 0), prediction.get('height', 0)])
            number_plate_region = image[y-25:y + h-20, x-80:x + w-80]
            number_plate_text = pytesseract.image_to_string(number_plate_region)
            detected.append(number_plate_text)

    if detected:
        print(detected)

    image = annotator.annotate(
        scene=image, detections=sv.Detections.from_roboflow(predictions)
    )

    cv2.imshow("Prediction", image)
    cv2.waitKey(1)

def process_frame(frame):
    inference.predict(
        model="anpr-debcj/1",
        image=frame,
        output_channel_order="BGR",
        api_key="dy1i7zKkh7CHFrAO7GBU",
        on_prediction=render,
    )

def capture_frames():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        # Perform image processing in a separate thread
        threading.Thread(target=process_frame, args=(frame,), daemon=True).start()

        # Display the original frame without annotation in the main thread
        cv2.imshow("Original Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start capturing and processing frames
    capture_frames()
