import json
import threading
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import tensorflow as tf

from model import DilatedUnet
from losses import dice_coef

@tf.function 
def train_step(inputs, y_true):
    with tf.GradientTape() as tape:
        y_pred = dun(inputs)
        loss = dun.loss(y_true, y_pred)

    gradients = tape.gradient(loss, dun.trainable_variables)
    optim.apply_gradients(zip(gradients, dun.trainable_variables))

    train_loss(loss)
    # train_acc()

@tf.function
def test_step(inputs, y_true):
    y_pred = dun(inputs)
    t_loss = dun.loss(y_true, y_pred)

    test_loss(t_loss)


class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


@threadsafe_generator
def train_generator(df):
    while True:
        shuffle_indices = np.arange(len(df))
        shuffle_indices = np.random.permutation(shuffle_indices)
        
        for start in range(0, len(df), BATCH_SIZE):
            x_batch = []
            y_batch = []
            
            end = min(start + BATCH_SIZE, len(df))
            # ids_train_batch = df.iloc[shuffle_indices[start:end]]
            ids_train_batch = df[start:end]
            
            # for _id in ids_train_batch.values:
            for _id in ids_train_batch:
                img = cv2.imread('input/train_hq/{}.jpg'.format(_id))
                assert img is not None
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                
                mask_id = int(_id.split('color')[1])
                # mask = cv2.imread('input/train_masks/{}_mask.png'.format(_id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread('input/train_masks/{:04d}.png'.format(mask_id),cv2.IMREAD_GRAYSCALE)
                assert mask is not None
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = np.expand_dims(mask, axis=-1)
                assert mask.ndim == 3
                
                # === You can add data augmentations here. === #
                # if np.random.random() < 0.5:
                #     img, mask = img[:, ::-1, :], mask[..., ::-1, :]  # random horizontal flip
                
                x_batch.append(img)
                y_batch.append(mask)
            
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.
            
            yield x_batch, y_batch


if __name__ == '__main__':

    BATCH_SIZE = 2
    WIDTH = 640
    HEIGHT = 480

    with open('input/image_list.json', 'r') as f:
        df_train = json.load(f)

    ids_train = list(map(lambda s: s.split('.')[0], df_train['img']))
    ids_train, ids_valid = train_test_split(ids_train, test_size=.1)
    train_gen = train_generator(ids_train)

    input = tf.Variable(tf.random.normal((1,480,640,1)))
    
    # out = dilated_net(input)
    # print('out = ', out)

    dun = DilatedUnet(mode='cascade',
                    filters=32,
                    n_class=1)

    optim = tf.keras.optimizers.RMSprop(learning_rate=dun.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    EPOCHS = 100
    MAX_ITER = 20
    for epoch in range(EPOCHS):
        for iter in range(MAX_ITER):
            x_batch, y_batch = next(train_gen)

            train_step(x_batch, y_batch)
            # skip test_step

            exit()

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch, train_loss.result()))