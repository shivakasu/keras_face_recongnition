#-*- coding: utf-8 -*-
import random
from keras.layers import Input,merge,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.models import Model
from sklearn.cross_validation import train_test_split
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from load_face import load_train_orl,load_test_orl,load_train_pie,load_test_pie


model=None

class Dataset:
    def __init__(self, path_name):
        self.train_images = None
        self.train_labels = None
        self.valid_images = None
        self.valid_labels = None
        self.test_images  = None            
        self.test_labels  = None
        self.path_name    = path_name
        self.input_shape = None
    def load(self, img_rows = 64, img_cols = 64, 
             img_channels = 1, nb_classes = 68):
        images, labels = load_train_pie(self.path_name)
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.2, random_state = random.randint(0, 100))        
        test_images,test_labels = load_test_pie(self.path_name)
        #当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        #这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)            
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)            
            
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')
        
            train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
            test_labels = np_utils.to_categorical(test_labels, nb_classes)                        
        
            train_images = train_images.astype('float32')            
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')
            
            train_images /= 255
            valid_images /= 255
            test_images /= 255            
        
            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images  = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels  = test_labels
            
 

def identity_block(input_tensor, kernel_size, filters):
    dim_ordering = K.image_dim_ordering()
    nb_filter1, nb_filter2, nb_filter3 = filters
    if dim_ordering == 'tf':
        axis = 3
    else:
        axis = 1
        
    out = Conv2D(nb_filter1, (1, 1), data_format="channels_last")(input_tensor)
    out = BatchNormalization(axis=axis)(out)
    out = Activation('relu')(out)

    out = out = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                              data_format="channels_last")(out)
    out = BatchNormalization(axis=axis)(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter3, (1, 1), data_format="channels_last")(out)
    out = BatchNormalization(axis=axis)(out)

    out = merge([out, input_tensor], mode='sum')
    out = Activation('relu')(out)
    return out


def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        axis = 3
    else:
        axis = 1

    out = Conv2D(nb_filter1, (1, 1), strides=strides,data_format="channels_last")(input_tensor)
    out = BatchNormalization(axis=axis)(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',data_format="channels_last")(out)
    out = BatchNormalization(axis=axis)(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter3, (1, 1), data_format="channels_last")(out)
    out = BatchNormalization(axis=axis)(out)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,data_format="channels_last")(input_tensor)
    shortcut = BatchNormalization(axis=axis)(shortcut)

    out = merge([out, shortcut], mode='sum')
    out = Activation('relu')(out)
    return out


def get_resnet50(dataset):
    if K.image_dim_ordering() == 'tf':
        axis = 3
    else:
        axis = 1
    inp = Input(shape=dataset.input_shape)
    dim_ordering = K.image_dim_ordering()
    out = ZeroPadding2D((3, 3), data_format="channels_last")(inp)
    out = Conv2D(64, (7, 7), strides=(2, 2), data_format="channels_last")(out)
    out = BatchNormalization(axis=axis)(out)
    out = Activation('relu')(out)
    out = MaxPooling2D((3, 3), strides=(2, 2), data_format="channels_last")(out)

    out = conv_block(out, 3, [64, 64, 256], strides=(1, 1))
    out = identity_block(out, 3, [64, 64, 256])
    out = identity_block(out, 3, [64, 64, 256])

    out = conv_block(out, 3, [128, 128, 512])
    out = identity_block(out, 3, [128, 128, 512])
    out = identity_block(out, 3, [128, 128, 512])
    out = identity_block(out, 3, [128, 128, 512])

    out = conv_block(out, 3, [256, 256, 1024])
    out = identity_block(out, 3, [256, 256, 1024])
    out = identity_block(out, 3, [256, 256, 1024])
    out = identity_block(out, 3, [256, 256, 1024])
    out = identity_block(out, 3, [256, 256, 1024])
    out = identity_block(out, 3, [256, 256, 1024])

    out = conv_block(out, 3, [512, 512, 2048])
    out = identity_block(out, 3, [512, 512, 2048])
    out = identity_block(out, 3, [512, 512, 2048])

    out = AveragePooling2D((2, 2), data_format="channels_last")(out)
    out = Flatten()(out)
    out = Dense(68, activation='softmax')(out)

    model = Model(inp, out)
    model.summary()

    return model


    
def save_mod(file_path):
    model.save(file_path)

def load_mod(file_path):
    model = load_model(file_path)


def evaluate(dataset):
    score = model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    
    
#训练模型
def train(dataset, batch_size = 40, nb_epoch = 100):        
    sgd = SGD(lr = 0.01, decay = 1e-6, 
              momentum = 0.9, nesterov = True) #采用SGD+momentum的优化器进行训练，首先生成一个优化器对象  
    model.compile(loss='categorical_crossentropy',
                       optimizer=sgd,
                       metrics=['accuracy'])   #完成实际的模型配置工作
               
    model.fit(dataset.train_images,
                   dataset.train_labels,
                   batch_size = batch_size,
                   epochs = nb_epoch,
                   validation_data = (dataset.valid_images, dataset.valid_labels),
                   shuffle = True)


if __name__ == '__main__':
    dataset = Dataset('data/PIE dataset/')    
    dataset.load()
    
#train
    model = get_resnet50(dataset)
    sgd = SGD(lr = 0.01, decay = 1e-6, 
              momentum = 0.9, nesterov = True)
    model.compile(loss='categorical_crossentropy',
                       optimizer=sgd,
                       metrics=['accuracy'])
    model.load_weights('face_model_resnet50.h5')
#    train(dataset)
#    save_mod('face_model_pie.h5')
    
#test
#    load_mod(file_path = '123.h5')
    print("loaded")
    evaluate(dataset)

