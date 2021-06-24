from mask_detector_image import mask_detection, class_selection

category_index, image, boxes, scores, classes, num = mask_detection('frozen_graphs',
                                                               '/frozen_inference_graph.pb',
                                                               '/labelmap.pbtxt',
                                                               2,'image2.jpg')

class_selection(scores, classes, image, category_index, boxes)