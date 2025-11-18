#该程序将文件夹下的所有文件分割成四个角度并存入相应的该文件夹下新建的四个角度的文件夹内
import os
import cv2
import numpy as np

# I90,I45,I135,I0=img[0,:,:],img[1,:,:],img[2,:,:],img[3,:,:]#根据偏振相机四个像素位置对应到不同偏振角图片
def fenge2four(img_acq):
    image_origin = cv2.imread(img_acq,0)
    #获取图像大小
    imgsize = image_origin.shape
    img_acq2four = np.zeros((4, int(imgsize[0]/2), int(imgsize[1]/2)),dtype=np.uint8)#构建存储4张图片的数组
    a1 = np.arange(0, imgsize[0], 2)#行的偶数
    a2 = np.arange(0, imgsize[1], 2)#列的偶数
    a3 = np.arange(1, imgsize[0], 2)#行的奇数
    a4 = np.arange(1, imgsize[1], 2)#列的奇数
    # 将图片的偶数行，偶数列元素取出，并放在返回变量内
    img1 = image_origin[a1, :]
    img_acq2four[0,:,:] = img1[:, a2]
    img2 = image_origin[a1, :]
    img_acq2four[1,:,:] = img2[:, a4]
    img3 = image_origin[a3, :]
    img_acq2four[2,:,:] = img3[:, a2]
    img4 = image_origin[a3, :]
    img_acq2four[3,:,:] = img4[:, a4]
    return img_acq2four


def remove_common_part(str1, str2):
    # 如果str1比str2长，交换它们
    if len(str1) < len(str2):
        str1, str2 = str2, str1

    # 用较短字符串替换较长字符串中的相同部分
    return str1.replace(str2, '')

#遍历文件夹下所有文件
folder_path="E:\Project/firstArtical\data\yinLiaoPingZi"
# new_folder_path="E:\Project\machineviewpython\data\calib/3"
new_folder_path=folder_path
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对文件进行进一步处理
        img=fenge2four(file_path)
        result = remove_common_part(root, folder_path)
        print(file_path)
        angle= ['90','45','135','0']
        for num,item in zip(range(1,5),angle):
            # 保存图片
            save_path = new_folder_path + result + "/" + item + "/" + file
            # 如果文件夹不存在，先创建文件夹
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img[num-1, :, :])
            print(save_path)
