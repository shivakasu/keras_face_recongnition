# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
                featurewise_center = False,             #是否使输入数据去中心化（均值为0），
                samplewise_center  = False,             #是否使输入数据的每个样本均值为0
                featurewise_std_normalization = False,  #是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization  = False,  #是否将每个样本数据除以自身的标准差
                zca_whitening = False,                  #是否对输入数据施以ZCA白化
                rotation_range = 5,                    #数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range  = 0.1,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range = 0.1,               #同上，只不过这里是垂直
                horizontal_flip = False,                 #是否进行随机水平翻转
                vertical_flip = False)       


#for m in range(40):
#    for k in range(10):
#        numstr = "{0:d}".format(k+1);
#        filename="bmp2/s"+str(m+1)+"/"+str(k+1)+'.BMP';
#        img = load_img(filename)  # this is a PIL image, please replace to your own file path
#        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#    
#        i = 0
#    
#        for batch in datagen.flow(x,
#                                  batch_size=1,
#                                  save_to_dir='bmp2/s'+str(m+1)+'/',#生成后的图像保存路径
#                                  save_prefix=numstr,
#                                  save_format='BMP'):
#            i += 1
#            if i > 50:
#                break
            
            
from PIL import Image  
for k in range(99):
    print(k)
    numstr = "{0:d}".format(k);
    filename="pic/3/"+str(k)+'.jpg'
    filename2="pic/"+str(k)+'.jpg'
#    img = Image.open(filename)
#    out = img.resize((128, 128)) #resize image with
#    out.save(filename2)
    img = load_img(filename)  # this is a PIL image, please replace to your own file path
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + (128,128,3))  # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0

    for batch in datagen.flow(x,
                              batch_size=1,
                              save_to_dir='pic/test/3/',#生成后的图像保存路径
                              save_prefix=numstr,
                              save_format='jpg'):
        i += 1
        if i > 10:
            break
    i=0
    for batch in datagen.flow(x,
                              batch_size=1,
                              save_to_dir='pic/train/3/',#生成后的图像保存路径
                              save_prefix=numstr,
                              save_format='jpg'):
        i += 1
        if i > 90:
            break