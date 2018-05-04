from imagehandler import ImageHandler
import extractor as e
from copy import deepcopy
import filehandler
import augmenter as a
import ann
import detector

# Get data
image_handler = ImageHandler(0.0)
train_images, train_targets = image_handler.get_all_train_data()

# Augment data
a.add_invert(train_images, train_targets)

# Select feature extraction methods
method = [e.apply_isodata_threshold]
# Apply methods
train_images = e.extract(method, train_images, True, True, 40)

# Create classifier
a_nn = ann.ANN()
# Train classifier
a_nn.train(train_images, train_targets)

# Read image to perform detection on
detection_image = filehandler.read_detection_image(2)

# Perform sliding window method on image
detector = detector.Detector(detection_image, 1, 20)
windows = detector.sliding_window()

# Apply feature extraction methods to images containing letters
extracted_frames = e.extract(method, deepcopy(windows), True)

# Predict letters
predictions = a_nn.predict(extracted_frames)

# Show predictions
detector.show_results(predictions, 0.8)
