# This is a sample Python script.

# Author: Krishnaragavan
# Date: 27/5/2021
# Import packages

import cv2
import numpy as np
import tensorflow as tf
import sys
from utils import label_map_util
from utils import visualization_utils as vis_util
from playsound import playsound


def mask_detection(inference_path, frozen_graph_path, labelmap, number_of_classes, input):
    detection_graph = tf.Graph()
    # Name of the directory containing the object detection module we're using
    TRAINED_MODEL_DIR = inference_path

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = TRAINED_MODEL_DIR + frozen_graph_path
    print(PATH_TO_CKPT)
    # Path to label map file
    PATH_TO_LABELS = TRAINED_MODEL_DIR + labelmap

    # Number of classes the object detector can identify
    NUM_CLASSES = number_of_classes

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.

    print("> ====== Loading frozen graph into memory")

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
    img = input

    # Load the Tensorflow model into memory.
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    sess = tf.Session(graph=detection_graph)
    image = cv2.imread(img)
    image = cv2.resize(image, (600, 600))
    image_expanded = np.expand_dims(image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    return category_index, image, boxes, scores, classes, num


def class_selection(scores, classes, image, category_index, boxes):
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

        if 1 in new_classes:

            k = vis_util.visualize_boxes_and_labels_on_image_array(image,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8,
                                                                   min_score_thresh=0.75)

            cv2.imshow('Object detector', k)
            cv2.imwrite('mask_detection_2.jpg', k)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif 2 in new_classes:

            k = vis_util.visualize_boxes_and_labels_on_image_array(image,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8,
                                                                   min_score_thresh=0.75)

            cv2.putText(image, "ALERT", (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow('Object detector', k)
            playsound(r"alert.wav")
            cv2.imwrite('mask_detection_2.jpg', k)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print("Here, There is no detection related to mask or not mask")

    except Exception as e:
        print("Errors", e)
