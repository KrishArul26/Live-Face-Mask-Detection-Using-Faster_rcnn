######## Webcam Object Detection Using Tensorflow-trained Classifier #########

# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from playsound import playsound
detection_graph = tf.Graph()


# Import utilites
# from utils import label_map_util

# Name of the directory containing the object detection module we're using
TRAINED_MODEL_DIR = 'frozen_graphs'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
print(PATH_TO_CKPT)
# Path to label map file
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'

# Number of classes the object detector can identify

NUM_CLASSES = 2
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes = NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.

print("> ====== Loading frozen graph into memory")
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)

while True:

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # single-column array, where each item in the column has the pixel RGB value
    _, frame = video.read()
    frame = cv2.resize(frame, (900, 900), interpolation=cv2.INTER_AREA)

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    result = scores.flatten()
    new_classes = classes.flatten()

    index = []

    for i, value in enumerate(result):
        if value > 0.70:
            index.append(i)

    new_class = []
    for ind in index:
        new_class.append(new_classes[ind])

    try:
        if 1 in new_class and 2 not in new_class:

            k = vis_util.visualize_boxes_and_labels_on_image_array(frame,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8,
                                                                   min_score_thresh=0.75)

            cv2.imshow('Object detector', k)

        elif 2 in new_class:

            k1 = vis_util.visualize_boxes_and_labels_on_image_array(frame,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8,
                                                                   min_score_thresh=0.75)

            cv2.putText(frame, "ALERT", (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            playsound(r"alert.wav")
            #cv2.resize(frame, (600, 600), interpolation=cv2.INTER_AREA)
            cv2.imshow('Object detector', k1)

        else:
            print("Here, There is no detection related to mask or not mask")

    except Exception as e:
        print("Errors", e)


    if cv2.waitKey(1) == ord('q'):
        break
# Clean up
video.release()
cv2.destroyAllWindows()


