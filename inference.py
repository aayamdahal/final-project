"""Handwritten math expression recognition.

Recognition logic is lifted verbatim from utils.py:identify_and_evaluate so the
model behaves identically; the only changes are: input is image bytes (decoded
in-memory, no filesystem round-trip), output is a structured dict, and the heavy
TensorFlow/Keras imports + model load are lazy so the pure helper
classes_to_expression stays importable and testable without TensorFlow.
"""
import os

# Class index -> symbol. 0-9 are digits; 10='-', 11='+', 12='*' (from original).
_OPERATORS = {10: '-', 11: '+', 12: '*'}

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

_model = None


def classes_to_expression(class_ids):
    """Map a list of predicted class indices to an expression string.

    Pure function, no TensorFlow dependency, so it is unit-testable on any host.
    """
    out = ''
    for c in class_ids:
        c = int(c)
        if c < 10:
            out += str(c)
        elif c in _OPERATORS:
            out += _OPERATORS[c]
    return out


def _load_model():
    """Load the Keras model once. Imports TF/Keras lazily (TF 2.1 / Keras 2.3)."""
    import tensorflow as tf
    from keras.models import model_from_json
    import keras.backend as K
    import keras.backend.tensorflow_backend as tfback

    K.set_image_data_format('channels_first')

    # Compatibility shim for TF 2.1 + Keras 2.3 GPU listing (from original utils.py).
    def _get_available_gpus():
        if tfback._LOCAL_DEVICES is None:
            devices = tf.config.list_logical_devices()
            tfback._LOCAL_DEVICES = [x.name for x in devices]
        return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

    tfback._get_available_gpus = _get_available_gpus

    with open(os.path.join(_MODEL_DIR, 'model_final.json'), 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(os.path.join(_MODEL_DIR, 'model_final.h5'))
    return model


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _segment_characters(img):
    """Return a list of (1,1,28,28) arrays, left-to-right. Lifted from utils.py."""
    import cv2
    import numpy as np

    blur = cv2.GaussianBlur(img, (25, 25), 0)
    ret, thresh = cv2.threshold(blur, 70, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ctrs, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    rects = [list(cv2.boundingRect(c)) for c in cnt]

    bool_rect = []
    for r in rects:
        l = []
        for rec in rects:
            flag = 0
            if rec != r:
                if (r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10)
                        and r[1] < (rec[1] + rec[3] + 10)
                        and rec[1] < (r[1] + r[3] + 10)):
                    flag = 1
                l.append(flag)
            if rec == r:
                l.append(0)
        bool_rect.append(l)

    dump_rect = []
    for i in range(0, len(cnt)):
        for j in range(0, len(cnt)):
            if bool_rect[i][j] == 1:
                area1 = rects[i][2] * rects[i][3]
                area2 = rects[j][2] * rects[j][3]
                if area1 == min(area1, area2):
                    dump_rect.append(rects[i])

    final_rect = [i for i in rects if i not in dump_rect]

    train_data = []
    for r in final_rect:
        x, y, w, h = r
        im_crop = thresh[y:y + h + 24, x:x + w + 24]
        im_resize = cv2.resize(im_crop, (28, 28))
        im_resize = np.reshape(im_resize, (1, 1, 28, 28))
        train_data.append(im_resize)
    return train_data


def evaluate_image(image_bytes):
    """Decode PNG/JPEG bytes, recognize the expression, evaluate it.

    Returns {"expression": str, "result": number}.
    Raises ValueError if nothing recognizable was drawn.
    """
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image.")

    chars = _segment_characters(img)
    if not chars:
        raise ValueError("No expression detected — try drawing more clearly.")

    model = _get_model()
    class_ids = []
    for ch in chars:
        ch = np.array(ch).reshape(1, 1, 28, 28)
        result = model.predict_classes(ch)
        class_ids.append(int(result[0]))

    expression = classes_to_expression(class_ids)
    if not expression:
        raise ValueError("Could not read any symbols.")

    value = eval(expression)  # bounded to digits and + - *
    return {"expression": expression, "result": value}
