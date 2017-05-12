# keras_face_recongnition  
  
注：  
1、我做的只是基础的cnn图像分类，并不是真正意义上的人脸识别。人脸识别技术参考[这篇知乎文章](https://zhuanlan.zhihu.com/shanren7)  
2、代码被我改的一团糟，有时间再整理吧  
3、opencv自带的人脸检测太渣了  


catch_vio.py：获取计算机摄像头源  
face_detect.py：在摄像头框里进行人脸检测  
get_picture.py：人脸检测的同时保存人脸图片为样本  
data_aug：对原始数据集进行数据增强  
load_face：加载数据集  
face_train：模型训练及测试  
face_train_resnet50.py：在resnet50上训练和测试PIE数据集  

  

face_model_ORL.h5：ORL数据集训练参数，图片尺寸92x112，batch_size = 40, epochs = 40,    正确率99.06%
face_model_PIE.h5：PIE数据集训练参数，图片尺寸64x64， batch_size = 40, epochs = 100,   正确率98.76%  
face_model_resnet50.h5：PIE数据集训练参数，图片尺寸64x64， batch_size = 40, epochs = 100,   正确率99.38%  
face_model_re.h5：我们寝室人脸数据集训练参数，图片尺寸128x128， batch_size = 50, epochs = 40,   正确率100%  
文件地址：[face-weights](http://pan.baidu.com/s/1pKKn5wR)


ORL_log：ORL数据集训练及测试输出信息  
PIE_log：PIE数据集训练及测试输出信息  
PIE_log_resnet50：PIE数据集训练及测试输出信息,训练过程被控制台覆盖找不回来了。。。  
real_time_log：我们寝室几个人的头像做数据集的训练及测试输出信息