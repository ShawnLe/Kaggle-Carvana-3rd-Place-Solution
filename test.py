import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from losses import (
    binary_crossentropy,
    dice_loss,
    bce_dice_loss,
    dice_coef,
    weighted_bce_dice_loss
)

# get all backends. Link: https://stackoverflow.com/questions/5091993/list-of-all-available-matplotlib-backends
# webagg works for some people. Link: https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/635139
# tried: Qt4Agg, Qt5Agg, agg, WX, WebAgg, tkAgg (windows default) 
# plt.switch_backend('Qt5Agg')
print('current backend is ', plt.get_backend())

model = load_model('model_weights.hdf5', custom_objects={'bce_dice_loss' : bce_dice_loss, 'dice_coef' : dice_coef})

max_try = 50
max_img_id = 1186
for i in range(max_try):
    
    _id = np.random.randint(max_img_id)
    img_file = 'input/train_hq/color{}.jpg'.format(_id)
    img = cv2.imread(img_file)    

    if img is not None:
        # doesn't work on windows
        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)

        fig = plt.figure()
        plt.get_current_fig_manager().window.state('zoomed') # works on Windows
        a = fig.add_subplot(1,2,1)
        rgb = img
        rgbplot = plt.imshow(rgb)
        a.set_title('rgb')

        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        pred = np.squeeze(pred, axis=0)
        pred = cv2.normalize(pred, cv2.NORM_MINMAX)

        a = fig.add_subplot(1,2,2)
        segplot = plt.imshow(pred)
        plt.show()
        a.set_title('segment')