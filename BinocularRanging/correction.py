
import cv2
import numpy
import numpy as np
from BinocularRanging import stereoconfig as stereoconfig
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Rectify(object):

    def getRectifyTransform(self,height, width, config):
        # 读取内参和外参
        left_K = config.cam_matrix_left
        right_K = config.cam_matrix_right
        left_distortion = config.distortion_l
        right_distortion = config.distortion_r
        R = config.R
        T = config.T
        # print(height,width)
        # 计算校正变换
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                          (width, height), R, T, alpha=0)
        map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
        return map1x, map1y, map2x, map2y, Q

    # 消除畸变
    def undistortion(self,image, camera_matrix, dist_coeff):
        undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

        return undistortion_image


    # 畸变校正和立体校正
    def rectifyImage(self,image1, image2, map1x, map1y, map2x, map2y):
        rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA,cv2.BORDER_CONSTANT)
        rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA,cv2.BORDER_CONSTANT)
        frame0 = cv2.resize(rectifyed_img1, None, fx=0.7, fy=0.7)
        frame1 = cv2.resize(rectifyed_img2, None, fx=0.7, fy=0.7)
        return rectifyed_img1, rectifyed_img2


    # 立体校正检验----画线
    def draw_line(self,image1, image2):
        # 建立输出图像
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1] + image2.shape[1]

        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:image1.shape[0], 0:image1.shape[1]] = image1
        output[0:image2.shape[0], image1.shape[1]:] = image2

        for k in range(15):
            cv2.line(output, (0, 50 * (k + 1)), (2 * width, 50 * (k + 1)), (0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)  # 直线间隔：100

        return output


    def start(self,imgl,imgr):
        iml = imgl
        imr = imgr
        height, width = iml.shape[0:2]
        #读取相机内外参数
        config = stereoconfig.stereoCamera()
        iml = self.undistortion(iml,config.cam_matrix_left,config.distortion_l)
        imr = self.undistortion(imr,config.cam_matrix_right,config.distortion_r)
        #立体矫正
        map1x, map1y, map2x, map2y, Q = self.getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        iml_rectified, imr_rectified = self.rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
        # print(Q[2][3])#焦距
        # cv2.imwrite("img/left.jpg",iml_rectified)
        # cv2.imwrite("img/right.jpg",imr_rectified)
        # 绘制等间距平行线，检查立体校正的效果
        # line = self.draw_line(iml_rectified, imr_rectified)
        # cv2.imshow("test",line)
        # cv2.waitKey(0)
        # print(Q[2][3])
        return iml_rectified,imr_rectified,Q

# if __name__ == '__main__':
#     r = Rectify()
#     iml = cv2.imread("img/l.jpg")
#     imr = cv2.imread("img/r.jpg")
#     r.start(numpy.array(iml),numpy.array(imr))



