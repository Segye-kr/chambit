import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from model import run_delf

def save_data():
    #변환할 이미지 목록 불러오기
    image_path = '/content/drive/MyDrive/참빛설계/data/unzipDataset/'

    # img_list = os.listdir(image_path) #디렉토리 내 모든 파일 불러오기
    # img_list_jpg = [img for img in img_list if img.endswith(".jpg")] #지정된 확장자만 필터링

    # print ("img_list_jpg: {}".format(img_list_jpg))
    gps = scipy.io.loadmat('/content/drive/MyDrive/참빛설계/data/GPS_Long_Lat_Compass.mat')
    gps_compass = gps['GPS_Compass']
    florida_idx=np.where(gps_compass[:,0]<=32.5)[0]

    img_list_np = []
    # print(len(img_list_jpg))
    cnt=0
    keys = []
    k=1
    # 파일 이름도 저장해야함
    print(cnt)
    for i in os.listdir(image_path):
        key = int(i.split('_')[0].lstrip('0'))
        cnt+=1
        if key not in florida_idx:
            continue 
        img = image.load_img(image_path+i, target_size=(224,224))
        # img = Image.open(image_path + i)
        # image = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
        img_array = image.img_to_array(img)
        val =img_array
        img_list_np.append(val) # X_train
        keys.append(key) # y_train
        
        if cnt%100 == 0:
            print(cnt)
    np_img = np.array(img_list_np)
    np.savez(f'/content/drive/MyDrive/참빛설계/data2/part.npz', image=np_img, keys=keys)

    k+=1
        #print(img_list_np)


def load_data():
    data=[]
    # for i in range(1,14):
    data.append(np.load(f'Dataset/part.npz'))

    cnt=0
    k=0
    images = []
    labels = []
    for d in data:  
        for item in d['image']:
            images.append(item)
            cnt+=1
            if cnt%100 == 0:
                print(cnt)
        for item in d['keys']:
            labels.append(int(item))
            
    images = np.array(images)
    labels = np.array(labels)
    gps = scipy.io.loadmat('Dataset/GPS_Long_Lat_Compass.mat')
    gps_compass = gps['GPS_Compass']
    florida_idx=np.where(gps_compass[:,0]<=32.5)[0]

    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

    i=0
    results=[]
    print(len(labels))
    print(len(images))
    for img in images:
        delf_result = run_delf(img)
        results.append(delf_result)
        if i%100==0:
            print(i)
        i+=1

    results_np = np.asarray(results)

    