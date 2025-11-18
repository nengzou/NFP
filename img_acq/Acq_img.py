"""
请将此文件放在想要存储图片的文件夹，并将条纹图放在该文件夹下strippattern文件夹内
设置一次投图采集的照片张数，采集组数num_set，投图的张数num_p
#下面两个文件位置没用到
# file_proj="strippattern\\"
# file_save="acquisition_fold\\"
"""

import PySpin
import sys
import matplotlib.pyplot as plt
import os
import keyboard
import time
import cv2

NUM_IMAGES=1 #调用一次acquire_images函数拍几张图片。
##设置参数和待投图的路径与获取图片的存放路径
num_p=38    #投图的张数
num_set=1


#创建窗口，将窗口移动至投影仪，并全屏显示。
cv2.namedWindow('window_name', cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("window_name", 1920, 0)
cv2.setWindowProperty('window_name', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# # 投一张全白设置曝光时间
# print("投一张全白")
# file_bright = "strippattern\\" +  "18.bmp"
# idx = cv2.imread(file_bright)
# cv2.imshow("window_name",idx)
# key = cv2.waitKey(200)
# #投条纹图设置曝光时间
# set_ExposureTime=True
# strip_image_next=1
# print("按下S键开始投条纹图，按下1键结束设置而开始采集")
# while set_ExposureTime:
#     if keyboard.is_pressed('s'):
#         print("投一张条纹")
#         file_bright = "strippattern\\" + str(strip_image_next) +".bmp"
#         idx = cv2.imread(file_bright)
#         cv2.imshow("window_name", idx)
#         strip_image_next=strip_image_next+1
#         key = cv2.waitKey(200)
#     elif keyboard.is_pressed('1'):
#         set_ExposureTime=False


global continue_recording#继续记录GUI的标志位
continue_recording = True

def acquire_images(cam, nodemap, nodemap_tldevice,filename):
    """
    This function acquires and saves NUM_IMAGES=1 images from a device.
    """

    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # 获取节点对象指针
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # 获取节点对象
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        #从节点对象获取整数值
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # 给节点对象赋新值
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  开始采集图片
        cam.BeginAcquisition()
        print('Acquiring images...')


        # 获取-转换-保存图片
        # 后处理器生成
        processor = PySpin.ImageProcessor()

        # Set default image processor color processing method
        #
        # *** NOTES ***
        # By default, if no specific color processing algorithm is set, the image
        # processor will default to NEAREST_NEIGHBOR method.
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        for i in range(NUM_IMAGES):
            try:

                #  Retrieve next received image
                #
                #  *** NOTES ***
                #  Capturing an image houses images on the camera buffer. Trying
                #  to capture an image that does not exist will hang the camera.
                #
                #  *** LATER ***
                #  Once an image from the buffer is saved and/or no longer
                #  needed, the image must be released in order to keep the
                #  buffer from filling up.
                image_result = cam.GetNextImage(1000)

                #  Ensure image completion
                #
                #  *** NOTES ***
                #  Images can easily be checked for completion. This should be
                #  done whenever a complete image is expected or required.
                #  Further, check image status for a little more insight into
                #  why an image is incomplete.
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:

                    #  Print image information; height and width recorded in pixels
                    #
                    #  *** NOTES ***
                    #  Images have quite a bit of available metadata including
                    #  things such as CRC, image status, and offset values, to
                    #  name a few.
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))

                    #  Convert image to mono 8
                    #
                    #  *** NOTES ***
                    #  Images can be converted between pixel formats by using
                    #  the appropriate enumeration value. Unlike the original
                    #  image, the converted one does not need to be released as
                    #  it does not affect the camera buffer.
                    #
                    #  When converting images, color processing algorithm is an
                    #  optional parameter.
                    image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)

                    #图片保存
                    image_converted.Save(filename)
                    print('Image saved at %s' % filename)

                    #  释放图片
                    image_result.Release()
                    print('')

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def display_images(cam, nodemap, nodemap_tldevice):

    global continue_recording

    sNodemap = cam.GetTLStreamNodeMap()

    # Change bufferhandling mode to NewestOnly
    node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
    if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve entry node from enumeration node
    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    if not PySpin.IsReadable(node_newestonly):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve integer value from entry node
    node_newestonly_mode = node_newestonly.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

    print('*** IMAGE ACQUISITION ***\n')
    try:
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        #
        #  *** NOTES ***
        #  What happens when the camera begins acquiring images depends on the
        #  acquisition mode. Single frame captures only a single image, multi
        #  frame catures a set number of images, and continuous captures a
        #  continuous stream of images.
        #
        #  *** LATER ***
        #  Image acquisition must be ended when no more images are needed.
        cam.BeginAcquisition()

        print('Acquiring images...')

        #  Retrieve device serial number for filename
        #
        #  *** NOTES ***
        #  The device serial number is retrieved in order to keep cameras from
        #  overwriting one another. Grabbing image IDs could also accomplish
        #  this.
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        print("请移动标定板在合适位置后")
        print('按 1 键开始采集第 %d 组图片'% s_index)

        # Figure(1) is default so you can omit this line. Figure(0) will create a new window every time program hits this line
        fig = plt.figure(1)

        # Close the GUI when close event happens
        # fig.canvas.mpl_connect('close_event', handle_close)

        # Retrieve and display images
        while (continue_recording):
            try:

                #  Retrieve next received image
                #
                #  *** NOTES ***
                #  Capturing an image houses images on the camera buffer. Trying
                #  to capture an image that does not exist will hang the camera.
                #
                #  *** LATER ***
                #  Once an image from the buffer is saved and/or no longer
                #  needed, the image must be released in order to keep the
                #  buffer from filling up.

                image_result = cam.GetNextImage(1000)

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:

                    # Getting the image data as a numpy array
                    image_data = image_result.GetNDArray()

                    # Draws an image on the current figure
                    plt.imshow(image_data, cmap='gray')

                    # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
                    # Interval is in seconds.
                    plt.pause(0.001)

                    # Clear current reference of a figure. This will improve display speed significantly
                    plt.clf()

                    # If user presses enter, close the program
                    if keyboard.is_pressed('1'):
                        print('Program is closing...')

                        # Close figure
                        # plt.close('all')
                        # input('Done! Press Enter to exit...')
                        continue_recording = False

                        #  Release image
                #
                #  *** NOTES ***
                #  Images retrieved directly from the camera (i.e. non-converted
                #  images) need to be released in order to keep from filling the
                #  buffer.
                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True

#关闭窗口句柄
def handle_close():
    global continue_recording
    continue_recording = False
#打开窗口句柄
def handle_open():
    global continue_recording
    continue_recording = True
#启动相机
def run_single_camera(cam,filename):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.
    """
    try:
        result = True
        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        # 初始化相机
        cam.Init()
        # 获取相机节点图
        nodemap = cam.GetNodeMap()
        # 拍摄图片
        result &= acquire_images(cam, nodemap, nodemap_tldevice,filename)
        # 释放相机
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result
#启动GUI
def run_single_GUI(cam):

    try:
        result = True

        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images
        result &= display_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


#相机初始化
system = PySpin.System.GetInstance()#初始化实例
version = system.GetLibraryVersion()#系统版本？是spinnaker的版本吗
print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))


iface_list = system.GetInterfaces()#获取接口
num_interfaces = iface_list.GetSize()
print('Number of interfaces detected: %i' % num_interfaces)

cam_list = system.GetCameras()#相机列表，系统相机列表

num_cams = cam_list.GetSize()#相机数量

print('Number of cameras detected: %i' % num_cams)
#如果没有连接相机则销毁相机列表和接口并释放相机初始化
if num_cams == 0 or num_interfaces == 0:
    cam_list.Clear()
    iface_list.Clear()
    system.ReleaseInstance()
    print('Not enough cameras!')
#从接口中遍历出相机位置
for iface in iface_list:
    # Query interface 遍历出接口iface供函数使用
    print("遍历接口，iface对接口操作")
    del iface


#遍历相机
for i, cam in enumerate(cam_list):
    print('遍历相机对cam相机 %d进行操作...' % i)

    for s_index in range(1, num_set + 1):
        # 新建文件夹如果存在则不新建
        if not os.path.exists(str(s_index)):
            # 如果不存在则创建文件夹
            os.mkdir(str(s_index))

        ##投一张图-采集一张,先确定物体位置，位置正确按：1

        run_single_GUI(cam)

        for index in range(1, num_p + 1):
            file = "strippattern\\" + str(index) + ".bmp"
            idx = cv2.imread(file)
            cv2.imshow("window_name",idx)
            key = cv2.waitKey(200)
            filename = str(s_index) + "/" + str(index) + ".bmp"
            run_single_camera(cam,filename)

        handle_open()


    print('相机 %d 操作完成... \n' % i)
    del cam


#释放相机
cam_list.Clear()
iface_list.Clear()
system.ReleaseInstance()

#按下enter键推出程序
# input('Done! Press Enter to exit...')

