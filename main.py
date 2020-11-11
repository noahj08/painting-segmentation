import segmentation_models as sm
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import tensorflow_datasets as tfds

keras.backend.set_image_data_format('channels_last')
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
sm.set_framework('tf.keras')
models = {'unet': sm.Unet(BACKBONE, classes=3)}

model = models['unet']
model.compile('adam', 'categorical_crossentropy', metrics=[sm.metrics.iou_score])



def train_coco():
    BATCH_SIZE = 16
    train_ds, val_ds = tfds.load('coco', split=['train', 'test'], download=True, batch_size=BATCH_SIZE)
    train_ds = tf.data.Dataset.from_tensor_slices(list(train_ds))
    val_ds = tf.data.Dataset.from_tensor_slices(list(val_ds))
    #X_train, y_train, X_val, y_val = pickle.load(open(filename, 'rb+'))
    X_train, y_train = train_ds["image"], train_ds["label"]
    X_val, y_val = val_ds["image"], val_ds["label"]
    X_train = preprocess_input(X_train)
    #print(y_train[0])
    X_val = preprocess_input(X_val)
    hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=10, validation_data=(X_val, y_val))
    pickle.dump(hist.history, open('br_only_hist.pickle', 'wb+'))


def train(filename='br.pickle'):
    X_train, y_train, X_val, y_val = pickle.load(open(filename, 'rb+'))
    X_train = preprocess_input(X_train)
    print(X_train.shape)
    print(y_train[0])
    X_val = preprocess_input(X_val)
    hist = model.fit(x=X_train, y=y_train, batch_size=16, epochs=10, validation_data=(X_val, y_val))
    pickle.dump(hist.history, open('br_only_hist.pickle', 'wb+'))

def predict(filename = 'pic.jpg'):
    image = Image.open(filename).convert('RGB')
    image = np.asarray(image, dtype=np.float32)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)
    out = model(image, training=False)
    out = np.array(out[0]*255).astype(np.uint8)
    im = Image.fromarray(np.squeeze(out))
    im.save('dog_seg.png')


train_coco()
