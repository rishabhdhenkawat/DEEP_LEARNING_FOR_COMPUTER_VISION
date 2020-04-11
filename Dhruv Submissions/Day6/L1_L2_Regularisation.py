import argparse
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

from dag_lib.preprocessing import SimplePreprocessor
from dag_lib.datasets import SimpleDatasetLoader
import numpy as np


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
ap.add_argument('-n', '--neighbors', required=False, type=int, default=1,
                help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', required=False, type=int, default=-1,
                help='# of jobs for k-NN distance (-1 uses all available cores)')
args = vars(ap.parse_args())

# Get list of image paths
image_paths = list(paths.list_images(args['dataset']))

# Initialize SimplePreprocessor and SimpleDatasetLoader and load data and labels
print('[INFO]: Images loading....')
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)

# Reshape from (3000, 32, 32, 3) to (3000, 32*32*3=3072)
data = data.reshape((data.shape[0], 3072))

# Print information about memory consumption
print('[INFO]: Features Matrix: {:.1f}MB'.format(float(data.nbytes / 1024*1000.0)))

# Encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split data into training (75%) and testing (25%) data
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)



penalty=[None,"l1","l2"]

for p in penalty:
    print(f"[INFO] : Training model with {p} penalty")
    class_=SGDClassifier(penalty=p,random_state=42,loss="log")
    class_.fit(train_x, train_y)
    score=class_.score(test_x, test_y)
    print(f"[INFO] : Model Accuracy : {score*100} %")
