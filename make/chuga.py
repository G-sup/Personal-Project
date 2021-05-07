from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen = ImageDataGenerator(horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
# pred_datagen = ImageDataGenerator(rescale=1./255)

image_generator = ImageDataGenerator()    

# x_train = train_datagen.flow_from_directory('C:/Users/Admin/Desktop/anime/an',seed=104,target_size=(128, 128),batch_size=50000,save_to_dir='C:/Users/Admin/Desktop/anime/an1')#,subset="training")
x_train = image_generator.flow_from_directory('C:/Users/Admin/Desktop/anime/new_face',seed=104,target_size=(64, 64),batch_size=100000)#,subset="training")
# print(x_train[0][0])

np.save('C:/Users/Admin/Desktop/project_64_nf.npy', arr=x_train[0][0])