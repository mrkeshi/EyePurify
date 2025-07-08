import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

MARGIN = 0

class HighAccuracyFaceDetector:
    def __init__(self, min_confidence=0.95, scale_factor=1.125):
        self.detector = MTCNN()
        self.min_confidence = min_confidence
        self.scale_factor = scale_factor

    def process_image_file(self, image_path):
        print("[Step 1] خواندن تصویر و آماده‌سازی...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        original_height, original_width = image.shape[:2]
        new_width = int(original_width * self.scale_factor)
        new_height = int(original_height * self.scale_factor)
        image = cv2.resize(image, (new_width, new_height))

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("[Step 2] تشخیص چهره‌ها در تصویر...")
        detections = self.detector.detect_faces(rgb_image)

        coords = []
        for i, detection in enumerate(detections):
            confidence = detection['confidence']
            if confidence < self.min_confidence:
                continue

            x, y, w, h = detection['box']
            x -= MARGIN
            y -= MARGIN
            w += 2 * MARGIN
            h += 2 * MARGIN
            x = max(0, x)
            y = max(0, y)
            right = min(x + w, new_width)
            bottom = min(y + h, new_height)

            coords.append((y, right, bottom, x))  # top, right, bottom, left

            cv2.rectangle(image, (x, y), (right, bottom), (0, 255, 255), 2)

        box_height = 30 * len(coords) + 10
        black_box = np.zeros((box_height, image.shape[1], 3), dtype=np.uint8)
        for i, (top, right, bottom, left) in enumerate(coords):
            text = f"Face {i + 1}: ({int(left)},{int(top)}) - ({int(right)},{int(bottom)})"
            cv2.putText(black_box, text, (10, 30 * (i + 1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        final_image = cv2.vconcat([image, black_box])
        print("[Step 3] تکمیل پردازش تصویر و آماده‌سازی خروجی.")
        return coords, final_image
