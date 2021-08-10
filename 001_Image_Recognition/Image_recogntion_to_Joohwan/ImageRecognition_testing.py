import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import imagenet_utils
import time
from PIL import Image


mobile = tf.keras.applications.mobilenet.MobileNet()  ## Mobilenetv1
path='/home/nvidia/sd/IITP_2ndyear/SpecializedMobileNet-master/SpecializedMobileNet-master/imagenet/test/n01443537/'

file_name= 'ILSVRC2012_val_00004677.JPEG';
##img_resized = cv2.resize(img,(224,224))

#pruned_path = "mobilenet_0.h5" ## pruned model
#pruned_model = tf.keras.models.load_model(pruned_path)

original_path = "mobilenet_original.h5" ## mobilenet V2
original_model = tf.keras.models.load_model(original_path)

#img = cv2.imread(path + file_name);
img = Image.open(path+file_name);
#resized_img= cv2.resize(img,(224,224))
resized_img= img.resize((224,224))
final_image =  np.expand_dims(resized_img,axis =0)
final_image=tf.keras.applications.mobilenet.preprocess_input(final_image) 
kk=0;
for x in range(10):
    st = time.time()
    #predictions = mobile.predict(final_image)
    #predictions = pruned_model.predict(final_image)
    predictions = original_model.predict(final_image)
    endt= time.time() - st;
    print(endt)
    kk=endt+kk;
    results = imagenet_utils.decode_predictions(predictions)
    print(results)

print(kk/10)