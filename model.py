
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras import Model
# from tensorflow.keras.layers import 

from losses import bce_dice_loss

class DilatedUnet(Model):

    def __init__(self,
                # input_shape=(1920,1080,3),
                mode='cascade',
                filters=44,
                n_block=3,
                lr=.0001,
                loss=bce_dice_loss,
                n_class=1):

        super(DilatedUnet, self).__init__()
        self.mode = mode
        self.filters = filters
        self.n_block = n_block
        self.lr = lr
        self.loss = loss
        self.n_class = n_class
    
    def call(self, inputs):
        return self.dilated_net(inputs)
        
    def encoder(self, x, filters=44, n_block=3, kernel_size=(3,3), activation='relu'):
        skip = []
        for i in range(n_block):
            x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
            x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
            skip.append(x)
            x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        return x, skip

    def bottleneck(self, x, filters_bottleneck, mode='cascade', depth=6, kernel_size=(3,3), activation='relu'):
        dilated_layers = []
        if mode == 'cascade' : 
            for i in range(depth):
                x = Conv2D(filters_bottleneck, kernel_size, activation=activation, padding='same', dilation_rate=2**i)(x)
                dilated_layers.append(x)
            return add(dilated_layers)
        elif mode == 'parallel' : 
            for i in range(depth):            
                dilated_layers.append(Conv2D(filters_bottleneck, kernel_size, activation=activation, padding='same', dilation_rate=2**i)(x))
            return add(dilated_layers)

    def decoder(self, x, skip, filters, n_block=3, kernel_size=(3,3), activation='relu'):
        for i in reversed(range(n_block)):
            x = UpSampling2D(size=(2,2))(x)
            x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
            x = concatenate([skip[i], x])  # concatenate to the conv num_filter dimension
            x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
            x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        return x

    def dilated_net(self, inputs,
                    # input_shape=(1920,1080,3),
                    # mode='cascade',
                    # filters=44,
                    # n_block=3,
                    # lr=.0001,
                    # loss=bce_dice_loss,
                    # n_class=1
    ):
        mode = self.mode
        filters = self.filters
        n_block = self.n_block
        lr = self.lr 
        loss = self.loss
        n_class = self.n_class

        enc, skip = self.encoder(inputs, filters, n_block)
        bottle = self.bottleneck(enc, filters_bottleneck=filters * 2**n_block, mode=mode)
        dec = self.decoder(bottle, skip, filters, n_block)
        classify = Conv2D(n_class, (1,1), activation='sigmoid')(dec)

        return classify