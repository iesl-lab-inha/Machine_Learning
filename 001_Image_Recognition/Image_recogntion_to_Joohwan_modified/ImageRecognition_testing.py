import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import imagenet_utils
import time
from PIL import Image
from tensorflow.keras.preprocessing import image
import pdb


mobile = tf.keras.applications.mobilenet.MobileNet()  ## Mobilenetv1
mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()## Mobilenetv1
#path='D:\001_Work_IESL\001_Task@IESL\2020\0018_Image_Recognition_1stYear_by_Big_Data\Processed_shan\Image_recogntion_to_Joohwan_modified\Apollo\'

#file_name= 'ILSVRC2012_val_00000994.JPEG';
file_name= '000031.JPG';
## 31
 ## 03

##img_resized = cv2.resize(img,(224,224))

pruned_path = "mobilenet_0.h5" ## pruned model
pruned_model = tf.keras.models.load_model(pruned_path)

#original_path = "mobilenet_original.h5" ## mobilenet V2
#original_model = tf.keras.models.load_model(original_path)

#img = cv2.imread(path + file_name);
#img = Image.open(path+file_name);
####img = Image.open(file_name);
#resized_img= cv2.resize(img,(224,224))
#pdb.set_trace()
#resized_img= img.resize((224,224))
img = image.load_img(file_name, target_size = (224,224))

resized_img = image.img_to_array(img)
final_image =  np.expand_dims(resized_img,axis =0)
final_image=tf.keras.applications.mobilenet.preprocess_input(final_image)
kk=0;
for x in range(10):
    st = time.time()
    #predictions = mobile.predict(final_image)
    predictions = pruned_model.predict(final_image)
    #predictions = original_model.predict(final_image)
    endt= time.time() - st;
    print(endt)
    kk=endt+kk;
    results = imagenet_utils.decode_predictions(predictions)
    print(results)

print(kk/10)