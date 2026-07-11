import base64
import io
import json

import numpy as np
from PIL import Image
from ultralytics import YOLO

CLASS_NAMES = [
    'missing_hole', 'mouse_bite', 'open_circuit',
    'short', 'spur', 'spurious_copper'
]


def model_fn(model_dir):
    print("Ładuję model z:", model_dir)
    model = YOLO(f"{model_dir}/best.pt")
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return np.array(image)
    else:
        raise ValueError(f"Nieobsługiwany content type: {request_content_type}")


def predict_fn(input_data, model):
    results = model.predict(input_data, conf=0.25, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        detections.append({
            'class_name': CLASS_NAMES[class_id],
            'confidence': float(box.conf[0]),
            'bbox_xyxy': [float(x) for x in box.xyxy[0].tolist()],
        })

    return {'detections': detections, 'count': len(detections)}


def output_fn(prediction, accept):
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Nieobsługiwany accept type: {accept}")