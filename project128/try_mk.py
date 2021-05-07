import os
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, ReLU
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from keras.preprocessing import image

#생성자 모델을 만듭니다.
generator = Sequential()
generator.add(Dense(1024*4*4, input_dim=100, activation='relu'))
generator.add(Reshape((4, 4, 1024)))
generator.add(Conv2DTranspose(1024, kernel_size=2, strides=2))
# generator.add(BatchNormalization())
generator.add(Conv2DTranspose(512, kernel_size=2, strides=2))
# generator.add(BatchNormalization())
generator.add(Activation(ReLU()))
generator.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'))
# generator.add(BatchNormalization())
generator.add(Activation(ReLU()))
generator.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
# generator.add(BatchNormalization())
generator.add(Activation(ReLU()))
generator.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
generator.add(Activation(ReLU()))
generator.add(Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='tanh'))
generator.summary()



#판별자 모델을 만듭니다.
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding='same',input_shape=(128,128,3)))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
# discriminator.add(BatchNormalization())
discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
# discriminator.add(BatchNormalization())
discriminator.add(Conv2D(512, kernel_size=4, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
# discriminator.add(BatchNormalization())
discriminator.add(Conv2D(1024, kernel_size=4, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
# discriminator.add(BatchNormalization())
discriminator.add(Conv2D(1, kernel_size=4, strides=1))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Flatten())
discriminator.add(Dropout(0.5))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002,beta_1=0.5))
discriminator.trainable = False
discriminator.summary()

#생성자와 판별자 모델을 연결시키는 gan 모델을 만듭니다.
ginput = Input(shape=(100,))
dis_output = discriminator(generator(ginput))
gan = Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002,beta_1=0.5))
gan.summary()

#신경망을 실행시키는 함수를 만듭니다.
def gan_train(epoch, batch_size, saving_interval):

  x_train = np.load('C:/Users/Admin/Desktop/project_128_nf.npy')
  x_train = x_train.reshape(x_train.shape[0], 128, 128, 3).astype('float32')
  x_train = (x_train - 127.5) / 127.5 
  # 127.5를 빼준 뒤 127.5로 나누어 줌으로 인해 -1에서 1사이의 값으로 바뀌게 됩니다.

  true = np.ones((batch_size, 1))
  fake = np.zeros((batch_size, 1))

  for i in range(epoch):
          # 실제 데이터를 판별자에 입력하는 부분입니다.
          idx = np.random.randint(0, x_train.shape[0], batch_size)
          imgs = x_train[idx]
          d_loss_real = discriminator.train_on_batch(imgs, true)

          #가상 이미지를 판별자에 입력하는 부분입니다.
          noise = np.random.normal(0, 1, (batch_size, 100))
          gen_imgs = generator.predict(noise)
          d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

          #판별자와 생성자의 오차를 계산합니다.
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
          g_loss = gan.train_on_batch(noise, true)

          print('epoch:%d' % i, ' d_loss:%.4f' % d_loss, ' g_loss:%.4f' % g_loss)

        # 이부분은 중간 과정을 이미지로 저장해 주는 부분입니다. 본 장의 주요 내용과 관련이 없어
        # 소스코드만 첨부합니다. 만들어진 이미지들은 gan_images 폴더에 저장됩니다.
          if i % saving_interval == 0:
              #r, c = 5, 5
              noise = np.random.normal(0, 1, (25, 100))
              gen_imgs = generator.predict(noise)

            #   gan.load_weights('ganc_modu_nf128.h5')

              gan.save_weights('ganc_modu_nf128.h5')

              # Rescale images 0 - 1

              gen_imgs = 0.5 * gen_imgs + 0.5

            #   img = image.array_to_img(gen_imgs[], scale=False)
            #   img.save(os.path.join("C:/Users/Admin/Desktop/anime/ganimages/", 'generated_nf128_' + str(i) + '.png'))
            
              fig, axs = plt.subplots(5, 5)
              count = 0
              
              for j in range(5):
                  for k in range(5):
                      axs[j, k].imshow(gen_imgs[count, :, :])#scale=False)
                      axs[j, k].axis('off')
                      count += 1
              fig.savefig("C:/Users/Admin/Desktop/anime/ganimages/gan_modu_%d.png" % i)

gan_train(100000, 256, 200)