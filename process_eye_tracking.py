import pandas as pd
import os
import numpy as np
import cv2

def normalize_box(box):
    if len(box) == 4:
        x1, y1, x2, y2 = box
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        return (x_min, y_min, x_max, y_max)
    elif len(box) == 2:
        x, y = box
        return (x, y, x, y)
    else:
        raise ValueError(f"Box must have 2 or 4 elements, got {len(box)}")

def point_in_box(x, y, box):
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max

def process_eye_tracking_and_faces(input_excel_path, face_coords_list, output_dir, input_image_path=None):
    print("[EyeTracking] شروع خواندن داده‌های نگاه...")
    os.makedirs(output_dir, exist_ok=True)
    output_excel_dir = os.path.join(output_dir, "excel")
    os.makedirs(output_excel_dir, exist_ok=True)

    face_boxes = [normalize_box(box) for box in face_coords_list]

    df = pd.read_csv(input_excel_path, delimiter='\t', header=None)
    if df.shape[1] == 1:
        df = pd.read_csv(input_excel_path, delimiter=',', header=None)

    if df.shape[1] < 4:
        raise ValueError(f"داده‌ها باید حداقل ۴ ستون داشته باشند ولی {df.shape[1]} یافت شد.")

    data = df.iloc[1:].copy()
    data.columns = ['X', 'Y', 'Pupil', 'Timestamp'] + list(data.columns[4:])
    data = data.sort_values('Timestamp').reset_index(drop=True)
    data['TimeDiff'] = data['Timestamp'].shift(-1) - data['Timestamp']
    data['TimeDiff'] = data['TimeDiff'].fillna(0)

    face_indices = []
    count_inside = 0
    for idx, row in data.iterrows():
        x, y = float(row['X']), float(row['Y'])
        face_idx = -1
        for i, box in enumerate(face_boxes):
            if point_in_box(x, y, box):
                face_idx = i
                break
        if face_idx != -1:
            count_inside += 1
        face_indices.append(face_idx)
    data['FaceIndex'] = face_indices

    if count_inside == 0:
        print("[EyeTracking] هشدار: هیچ نقطه نگاهی داخل هیچ چهره‌ای نبود.")

    data_filtered = data[data['FaceIndex'] != -1]
    time_per_face = data_filtered.groupby('FaceIndex')['TimeDiff'].sum()

    n_faces = len(face_coords_list)
    output_rows = []
    for face_idx in range(n_faces):
        time_value = time_per_face.get(face_idx, 0)
        output_rows.append({
            'Face': f'Face {face_idx + 1}',
            'Time(ms)': time_value
        })

    total_time = time_per_face.sum()
    output_rows.append({
        'Face': 'Total',
        'Time(ms)': total_time
    })

    output_df = pd.DataFrame(output_rows)
    filename = os.path.basename(input_excel_path)
    filename_no_ext = os.path.splitext(filename)[0]
    output_csv_path = os.path.join(output_excel_dir, f"{filename_no_ext}_processed.csv")
    output_df.to_csv(output_csv_path, index=False)

    if input_image_path is not None:
        print("[EyeTracking] رسم مدت زمان نگاه روی تصویر...")
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"[EyeTracking] خطا: تصویر {input_image_path} خوانده نشد.")
        else:
            max_time = max(time_per_face.max(), 1)
            for i, box in enumerate(face_boxes):
                x_min, y_min, x_max, y_max = map(int, box)
                time_val = time_per_face.get(i, 0)
                intensity = int(255 * (time_val / max_time))
                color = (0, 0, intensity)  # رنگ قرمز با شدت متغیر

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                text = f"{time_val:.0f} ms"
                cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            output_image_dir = os.path.join(output_dir, "images")
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_path = os.path.join(output_image_dir, f"{filename_no_ext}_processed.png")
            cv2.imwrite(output_image_path, image)
            print(f"[EyeTracking] تصویر پردازش شده ذخیره شد: {output_image_path}")

    print(f"[EyeTracking] پردازش داده‌های نگاه و ذخیره فایل خروجی کامل شد: {output_csv_path}")
    return output_df
