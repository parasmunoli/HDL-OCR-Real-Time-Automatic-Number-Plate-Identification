import cv2
import inference
import supervision as sv
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

annotator = sv.BoxAnnotator()

def render(predictions, image):
    if isinstance(predictions, list):
        #print(type(predictions))
        return
    
    detected = []
    data = predictions['predictions']
    for prediction in data:
    # Check if the prediction is for the 'Number Plate' class
        if prediction.get('class') == 'Number Plate' or prediction.get('class_id') == 0:
            # Extract coordinates of the bounding box
            x, y, w, h = map(int, [prediction.get('x', 0), prediction.get('y', 0), prediction.get('width', 0), prediction.get('height', 0)])
            #print(x,y,w,h)
            
            # Crop the region corresponding to the Number Plate
            number_plate_region = image[y-25:y + h-20, x-80:x + w-80]
            
            #cv2.rectangle(image, (x-80, y-25), (x + w-80, y + h-20), (0, 255, 0), 2)

            # Perform OCR on the cropped region using Tesseract
            number_plate_text = pytesseract.image_to_string(number_plate_region)
            
            # Append the detected number plate text to the list
            detected.append(number_plate_text)

    # Print Number plate details
    if detected != []:
        print(detected)
    
    #Camera on (Object detection)
    image = annotator.annotate(
        scene=image, detections=sv.Detections.from_roboflow(predictions)
    )

    # Display the annotated image(Realtime frame output)
    cv2.imshow("Prediction", image)
    cv2.waitKey(1)

#connecting to roboflow API
inference.Stream(
    source=0,
    model="anpr-debcj/1",
    output_channel_order="BGR",
    use_main_thread=True,
    api_key="****",  
    on_prediction=render,
)
