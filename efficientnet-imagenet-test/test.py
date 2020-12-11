import numpy as np
import os
from PIL import Image
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import decode_predictions, preprocess_input

from nltk.corpus import wordnet


model = EfficientNetB1(weights='imagenet')

# load in an image (224, 224, 3)
size = (240,240)


def getHypers(word):
    out = 'NONE'
    if word != 'NONE':
        r_net = wordnet.synsets(word)[0]
        try:
            out = r_net.hypernyms()[0].lemma_names()[0]
        except:
            pass

    return out

def printClasses(title, classes):
    num = 8
    dash = '-' * 200
    fmt = ''.join('{0[' + str(i) + ']:<28}' for i in range(num))
    print('')
    print(title)
    print(dash)
    topfmt = '{:<28}' * num
    print(topfmt.format('Original', 'h1', 'h2', 'h3','h4','h5','h6','h7'))
    print(dash)

    for r in classes:
        h = [r[1]]
        for i in range(num):
            h.append(getHypers(h[i]))

        print(fmt.format(h))

def detectImageClasses(filename):
    image = Image.open(filename).resize(size)
    x = np.array(image, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, data_format=None)
    preds = model.predict(x)
    #  (class_name, class_description, score)
    results = decode_predictions(preds, top=20)

    printClasses(filename, results[0])


for i in range(1,5):
    fname = str(i) + ".jpg"
    print(fname)
    detectImageClasses(fname)


# image = Image.open("ice.jpg")
# image = image.resize(size)
# x = np.array(image, dtype=np.float32)
# x = np.expand_dims(x, axis=0)

# x = preprocess_input(x, data_format=None)

# preds = model.predict(x)

# #  (class_name, class_description, score)
# results = decode_predictions(preds, top=20)


# image = Image.open("pika.jpg").resize(size)
# x = np.array(image, dtype=np.float32)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x, data_format=None)
# preds = model.predict(x)

# #  (class_name, class_description, score)
# results = decode_predictions(preds, top=20)


# print(' ')
# print('PIKA')
# print(dash)
# print('{:<30}{:<30}{:<30}{:<30}'.format('Original', 'h1', 'h2', 'h3'))
# print(dash)
# for r in results[0]:
#     r1 = getHypers(r[1])
#     r2 = getHypers(r1)
#     r3 = getHypers(r2)
#     print('{:<30}{:<30}{:<30}{:<30}'.format(r[1], r1, r2, r3))

# print('')

