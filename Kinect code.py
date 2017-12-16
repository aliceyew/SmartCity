#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      TaaSoratana
#
# Created:     11/10/2017
# Copyright:   (c) TaaSoratana 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from pykinect2 import PyKinectV2 as pk2
from pykinect2 import PyKinectRuntime
import cv2, numpy as np, os, time, math, ctypes

class kinect_images_get:

    def __init__(self):
        # Sensor Initialization
        self.sensor = PyKinectRuntime.PyKinectRuntime(pk2.FrameSourceTypes_Depth)

        # CV_image Initialization
        self.width = self.sensor.color_frame_desc.Width
        self.height = self.sensor.color_frame_desc.Height
        self.cv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # loop Initialization
        self.spin = True

    def get_rgb(self):
        if self.sensor.has_new_color_frame():
            self.cv_image = self.sensor.get_last_color_frame()
            self.cv_image  = np.reshape(self.cv_image, (self.height, self.width, -1))
            self.cv_image = cv2.cvtColor(self.cv_image,cv2.COLOR_BGRA2BGR)

        pass

    def get_depth(self, i):
        if self.sensor.has_new_depth_frame():
            depth_arr = self.sensor.get_last_depth_frame()
            #print(depth_arr)
            self.cv_image3 = np.reshape(depth_arr ,(424, 512))
            #print(i)
            #print(self.cv_image3)
            self.cv_image2 = self.cv_image3*255
            cv2.imshow("depth", self.cv_image2)

            self.depth_ptr = np.ctypeslib.as_ctypes(depth_arr.flatten())
            L = depth_arr.size
            CS = self.height*self.width
            CameraSpacePointArray = pk2._CameraSpacePoint*CS
            self.csp_arr = CameraSpacePointArray()
            error_state = self.sensor._mapper.MapColorFrameToCameraSpace(L, self.depth_ptr,
                CS, self.csp_arr)
            pts_float_csp = ctypes.cast(self.csp_arr, ctypes.POINTER(ctypes.c_float))
            self.depth_dat = np.copy(np.ctypeslib.as_array(pts_float_csp,
                shape=(self.height, self.width, 3)))
            print(self.depth_dat[self.y, self.x, 2])

            #print(self.depth_dat[:, :, 2])

            filename = "frame" + str(i) + ".txt"
            filename2 = "frame" + str(i) + ".JPG"

            np.savetxt(filename, self.depth_dat[:, :, 2], delimiter=" ", fmt = '%.4f')
            cv2.imwrite(filename2, self.cv_image2)

            # f = open(filename, 'w')
            # for i in range(self.depth_dat.shape[0]):
            #     for j in range(self.depth_dat.shape[1]):
            #         f.write(str(self.depth_dat[i, j, 2]))
            #         f.write(" ")
            #     f.write("\n")
            # f.close()

    def extract_coordinate_color_val(self, x, y):
        self.x = x
        self.y = y
        #print('Pixel value at coordinate (%d, %d) = %s') % (self.x,self.y, self.cv_image[self.y,self.x,:])
        cv2.circle(self.cv_image,(self.x,self.y), 8, (0,0,255), 1)

    def run(self):
        try:
            i = 5000
            while self.spin:
                cv2.imshow('Image', self.cv_image)

                self.get_rgb()
                self.extract_coordinate_color_val(800, 400) # x, y
                self.get_depth(i)
                i += 1
                # Program breaker key input
                k = cv2.waitKey(1)
                if k == ord('q'): # q = Exit
                    self.spin = False

        except Exception as e:
            print(e)

        # Sensor deactivation
        self.sensor.close()


def main():

    lab1 = kinect_images_get()
    lab1.run()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
