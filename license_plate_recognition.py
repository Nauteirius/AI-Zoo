import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from ultralytics import YOLO

def crop_inner_plate(image, border=5):
    """
    Crops a margin around the license plate to remove outer borders or frames.
    :param image: Input plate image.
    :param border: Number of pixels to remove from each side.
    :return: Cropped image without frame.
    """
    h, w = image.shape[:2]
    return image[border:h - border, border:w - border]

def preprocess_plate_for_ocr(image):
    """
    Horizontally stretches the image and converts to RGB for better OCR performance.
    """
    # Stretch horizontally (eg.2x wider)
    resized = cv2.resize(image, None, fx=2.0, fy=1.0, interpolation=cv2.INTER_LINEAR)

    # Convert to RGB if needed
    if len(resized.shape) == 2:
        prepped = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        prepped = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return prepped

def show_candidates(candidates):
    for i, (plate, text, conf) in enumerate(candidates):
        plt.figure()
        plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        plt.title(f"Plate #{i} → '{text}' ({round(conf * 100)}%)")
        plt.axis("off")
    plt.show()

def detect_license_plate_from_car_boxes(image):
    """
    Detects cars using YOLOv8, then applies OCR within the car region to find license plates.
    Returns cropped images with recognized text and confidence.
    """
    model = YOLO("yolov8n.pt")
    results = model.predict(image)[0]
    names = model.model.names

    reader = easyocr.Reader(['en'], gpu=False)
    detected_plates = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        print(f"Detected class: {class_name}")

        if class_name in ["car","bus","truck","motorcycle"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            car_crop = image[y1:y2, x1:x2]

            # Run OCR inside car region
            ocr_results = reader.readtext(car_crop)

            for (bbox, text, conf) in ocr_results:
                x_coords = [int(pt[0]) for pt in bbox]
                y_coords = [int(pt[1]) for pt in bbox]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                plate_crop_raw = car_crop[y_min:y_max, x_min:x_max]
                
                # 🟢 preprocessing before OCR
                # Crop frame/margin from plate
                plate_cropped = crop_inner_plate(plate_crop_raw, border=5)

                # Preprocess (horizontal stretch, convert to RGB)
                plate_preprocessed = preprocess_plate_for_ocr(plate_cropped)

                # OCR
                plate_text = reader.readtext(plate_preprocessed, detail=0)

           
                plate_text = reader.readtext(plate_cropped, detail=0)

                if plate_text:
                    detected_plates.append((plate_cropped, plate_text[0].strip(), conf))

    return detected_plates

if __name__ == "__main__":
    image_path = "sample/car_plate.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Failed to load image.")
    else:
        plates = detect_license_plate_from_car_boxes(image)
        print(f"✅ License plate candidates found: {len(plates)}")

        for i, (plate_img, text, conf) in enumerate(plates):
            print(f"🔍 Plate #{i + 1}: '{text}' ({round(conf * 100)}%)")

        show_candidates(plates)
