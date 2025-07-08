import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
from face_detector import HighAccuracyFaceDetector
from process_eye_tracking import process_eye_tracking_and_faces

if __name__ == "__main__":
    input_dir = "PrimaryData/images"
    output_dir = "GeneratedData/images"
    os.makedirs(output_dir, exist_ok=True)

    detector = HighAccuracyFaceDetector(min_confidence=0.95, scale_factor=1.125)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in image_files:
        print(f"[Main] شروع پردازش تصویر: {filename}")
        input_path = os.path.join(input_dir, filename)
        coords, processed_img = detector.process_image_file(input_path)
        print(f"[Main] یافتن {len(coords)} چهره در تصویر {filename}")

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_img)
        print(f"[Main] تصویر پردازش شده ذخیره شد: {output_path}")

        excel_path = os.path.join("PrimaryData/excel", os.path.splitext(filename)[0] + ".csv")

        if os.path.exists(excel_path):
            face_coords_fixed = [(left, top, right, bottom) for (top, right, bottom, left) in coords]
            print(f"[Main] شروع پردازش داده‌های نگاه برای {excel_path} ...")
            summary_df = process_eye_tracking_and_faces(excel_path, face_coords_fixed, "GeneratedData", input_image_path=output_path)
            print(f"[Main] پردازش داده‌های نگاه برای {filename} تکمیل شد.")
        else:
            print(f"[Main] داده‌های نگاه برای {filename} یافت نشد: {excel_path}")
