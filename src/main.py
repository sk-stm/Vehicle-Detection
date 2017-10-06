from moviepy.editor import VideoFileClip
import glob
from src.DataSet import DataSet
from src.classifier import Classifier
from src.efficient_car_finder import EfficientCarFinder
import os
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

# define Video File to use
VIDEO_PATH = '../project_video.mp4'
VIDEO_OUTPUT = '../output_images/output.mp4'
TEST_IMG_PATH = '../test_images/test*.jpg'

colorspace = 'HSV'
orient = 13
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (16, 16)
hist_bins = 16
spatial_feat = False
hist_feat = False
hog_feat = True
# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
# we do 16x16 = 169
window = 64

class ImageProcessor:

    def process_image(self, img):
        """
        Detect lanes in the image.
        :param img: the image to detect lanes in
        :return: the image with detected lanes in
        """


def train_classifier():
    if not os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data.p")):
        dataset = DataSet(colorspace, orient, pix_per_cell, hog_channel, spatial_size,
                          hist_bins, spatial_feat, hist_feat, hog_feat)
        X_train, y_train, X_scaler = dataset.prepare_data_set()
        pickle.dump([X_train, y_train, X_scaler], open("data.p", "wb"))
    else:
        X_train, y_train, X_scaler = pickle.load(open("data.p", "rb"))

    if not os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "classifier.p")):
        cls = Classifier()
        cls.train(X_train, y_train)
        pickle.dump(cls, open("classifier.p", "wb"))
    else:
        cls = pickle.load(open("classifier.p", "rb"))

    return cls, X_scaler


def main():
    """
    Main method of the programm.
    :return:
    """
    img_proc = ImageProcessor()

    # get test images
    test_imgs = glob.glob(TEST_IMG_PATH)

    # get trained classifier
    cls, X_scaler = train_classifier()

    y_start_stop = [(400, 500), (400, 600), (400, 656), (400, 656)]
    ecf = EfficientCarFinder()

    # run on test images
    for img_path in test_imgs:
        pred_bboxs = []
        img = mpimg.imread(img_path)
        for i, scale in enumerate([1, 1.5, 2, 2.5, 3, 4]):
            bboxs = ecf.find_cars(img, y_start_stop[i][0], y_start_stop[i][1], scale, cls.classifier,
                                  X_scaler, spatial_size, hist_bins, window, orient=orient, pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block)
            pred_bboxs.append(bboxs)

        draw_img = np.copy(img)
        for bbx_per_scale in pred_bboxs:
            for bbx in bbx_per_scale:
                cv2.rectangle(draw_img, (bbx[0][0], bbx[0][1]), (bbx[1][0], bbx[1][1]), (0, 0, 255), 6)

        plt.imshow(draw_img)
        plt.show()

    pickle.dump(pred_bboxs, open("bbox_pickle.p", "wb"))

    # # detect lanes in the video file
    # white_output = VIDEO_OUTPUT
    # clip1 = VideoFileClip(VIDEO_PATH)
    # white_clip = clip1.fl_image(img_proc.process_image)
    # white_clip.write_videofile(white_output, audio=False)

if __name__ == '__main__':
    # initialize camera undistortion

    # run
    main()

