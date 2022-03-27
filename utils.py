import cv2
import numpy
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import keras.backend as K
K.set_image_data_format('channels_first')
from keras.models import model_from_json
import keras.backend.tensorflow_backend as tfback
import cv2
import numpy as np
from extract_answers import extract_box

def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


json_file = open('model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_final.h5")


def identify_and_evaluate(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    if img is not None:
        blur = cv2.GaussianBlur(img, (25, 25), 0)
        ret, thresh = cv2.threshold(blur, 70, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # ret,thresh=cv2.threshold(img,70,255,cv2.THRESH_BINARY)
        ctrs, h = cv2.findContours(thresh, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

        w = int(28)
        h = int(28)
        train_data = []
        rects = []
        for c in cnt :
            x, y, w, h = cv2.boundingRect(c)
            rect = [x, y, w, h]
            rects.append(rect)

        bool_rect = []
        for r in rects:
            l = []
            for rec in rects:
                flag = 0
                if rec != r:
                    if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (rec[1] + rec[3] + 10) and rec[1] < (r[1] + r[3] + 10):
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
                    if(area1 == min(area1,area2)):
                        dump_rect.append(rects[i])

        final_rect = [i for i in rects if i not in dump_rect]
        for r in final_rect:
            x = r[0]
            y = r[1]
            w = r[2]
            h = r[3]
            # im_crop =thresh[y:y+h+10,x:x+w+10]
            im_crop = thresh[y:y + h + 24, x:x + w + 24]


            im_resize = cv2.resize(im_crop, (28, 28))
            im_resize = np.reshape(im_resize, (1, 28, 28))
            train_data.append(im_resize)

    final_res = ''
    final_str = ''
    for i in range(len(train_data)):
        train_data[i] = np.array(train_data[i])
        train_data[i] = train_data[i].reshape(1, 1, 28, 28)
        result = loaded_model.predict_classes(train_data[i])

        if result[0] < 10:
            final_res += str(result[0])
            final_str += str(result[0])
        else:
            if result[0] == 10:
                final_res = final_res + '-'
                final_str += '-'
            if result[0] == 11:
                final_res = final_res + '+'
                final_str += '+'
            if result[0] == 12:
                final_res = final_res + '*'
                final_str += '*'

    return (f'{final_str} = {eval(final_res)}')


if __name__ == '__main__':
    print(identify_and_evaluate('.\\test_images\\4p3.jpg'))


