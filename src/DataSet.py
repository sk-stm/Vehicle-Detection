import glob
from sklearn.preprocessing import StandardScaler
from src.lesson_functions import *

class DataSet():

    def __init__(self, colorspace='HSV', orient=13, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                 spatial_size=(16, 16), hist_bins=16, spatial_feat=False, hist_feat=False, hog_feat=True):
        self. colorspave = colorspace
        self.orient = orient
        self. pix_per_cell = pix_per_cell
        self.scell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

    def prepare_data_set(self):
        # Divide up into cars and notcars
        cars = glob.glob('../vehicles/*/*.png')
        notcars = glob.glob('../non-vehicles/*/*.png')

        car_features = extract_features(cars, color_space=self.colorspace,
                                        spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                        orient=self.orient, pix_per_cell=self.pix_per_cell,
                                        cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        notcar_features = extract_features(notcars, color_space=self.colorspace,
                                           spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                           orient=self.orient, pix_per_cell=self.pix_per_cell,
                                           cell_per_block=self.cell_per_block,
                                           hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                           hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        return scaled_X, y, X_scaler
