TRACKING_ALGORITHMS = {
    "Strongsort": "strongsort",
    "Deepsort": "deepsort",
}

CLASS_NAMES = {
    0: "Pedestrian",
    1: "People",
    2: "Bicycle",
    3: "Car",
    4: "Van",
    5: "Truck",
    6: "Tricycle",
    7: "Awning-Tricycle",
    8: "Bus",
    9: "Motor",
    10: "Others",
}

CUSTOM_YOLOV8M = "Custom YoloV8m"
CUSTOM_YOLOV8X = "Custom YoloV8x"
CUSTOM_YOLOV5M = "Custom YoloV5m"

DETECTION_MODELS_PATH = {
    CUSTOM_YOLOV8M: "./weights/Mark5(v8m).pt",
    CUSTOM_YOLOV8X: "./weights/Mark1(v8x).pt",
    CUSTOM_YOLOV5M: "./weights/Mark3(v5m).pt"
}

DETECTION_MODELS = [
    CUSTOM_YOLOV8M,
    CUSTOM_YOLOV8X,
    CUSTOM_YOLOV5M
]

def get_tracking_algorithm_names():
    return TRACKING_ALGORITHMS.keys()

def get_tracking_algorithm_value(name):
    return TRACKING_ALGORITHMS[name]