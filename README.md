# keras_face_recongnition

catch_vio.py：获取计算机摄像头源  
face_detect.py：在摄像头框里进行人脸检测  
get_picture.py：人脸检测的同时保存人脸图片为样本
data_aug：对原始数据集进行数据增强
load_face：加载数据集
face_train：模型训练及测试

face_model.h5：ORL数据集训练参数，图片尺寸92x112，batch_size = 40, nb_epoch = 40
ORL_log：ORL数据集训练及测试输出信息