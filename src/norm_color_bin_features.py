import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob
from skimage.feature import hog

# TODO delete file

class NormColorBinFreature():

    def __init__(self):
        pass

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                      visualise=True, feature_vector=False)
            return features, hog_image
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                           visualise=False, feature_vector=feature_vec)
            return features

    # Define a function to compute binned color features
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features


    # Define a function to compute color histogram features
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the feature vector
        return hist_features


    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs, orient=13, pix_per_cell=8, cell_per_block=2, cspace='HSV', spatial_size=(32, 32),
                         hist_bins=32, hist_range=(0, 256)):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            else:
                feature_image = np.copy(image)
            # Apply bin_spatial() to get spatial color features
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            # Apply color_hist() also with a color space option now
            hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            # get hog features
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hog_features = self.get_hog_features(gray, orient=orient, pix_per_cell=pix_per_cell,
                                                 cell_per_block=cell_per_block)

            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        # Return list of feature vectors
        return features

if __name__ == "__main__":
    cars = glob.glob('../vehicles/*/*.png')
    notcars = glob.glob('../non-vehicles/*/*.png')
    ncbf = NormColorBinFreature()

    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    car_features = ncbf.extract_features(cars, cspace='HSV', spatial_size=(32, 32),
                                         hist_bins=32, hist_range=(0, 256), orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block)
    notcar_features = ncbf.extract_features(notcars, cspace='HSV', spatial_size=(32, 32),
                                            hist_bins=32, hist_range=(0, 256), orient=orient, pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block)

    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(cars))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.show()
    else:
        print('Your function only returns empty feature vectors...')