import glob
import time
from sklearn.preprocessing import StandardScaler
from src.lesson_functions import *
from src.classifier import Classifier
from sklearn.model_selection import train_test_split
import pickle
import os


# Divide up into cars and notcars
cars = glob.glob('../vehicles/*/*.png')
notcars = glob.glob('../non-vehicles/*/*.png')

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time

if not os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "train_data.p")):
    colorspace = 'HSV'
    orient = 13
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    car_features = extract_features(cars, color_space=colorspace,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=colorspace,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    t2 = time.time()
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    pickle.dump([X_train, X_test, y_train, y_test], open("train_data.p", "wb"))
else:
    X_train, X_test, y_train, y_test = pickle.load(open("train_data.p", "rb"))

print('Using:', orient,'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

t=time.time()
if not os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "classifier.p")):
    cls = Classifier()
    cls.train(X_train, y_train)
    svr = cls.classifier
    pickle.dump(cls, open("classifier.p", "wb"))
else:
    cls = pickle.load(open("classifier.p", "rb"))
    svr = cls.classifier
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svr.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svr.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
