from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def getImage_and_resize(path, new_width=224, new_height=224):
  input_data_image = tf.keras.utils.load_img(path, target_size=(224,224))
  input_data_array = tf.keras.utils.img_to_array(input_data_image)

  return input_data_image, input_data_array

def load_saved_delf_data():
  delf_ds = np.load(f'Dataset/delf_demo.npy', allow_pickle=True)
  labels = np.load(f'Dataset/delf_demo_labels.npy', allow_pickle=True)
  images = np.load(f'Dataset/image.npz')

  return delf_ds, labels, images['image']