from __future__ import print_function, division, generators, unicode_literals
from support import *
from face_detect import FaceDetect
from process_results import Results
import classifier_gcam
#import cv2

# TODO: return values for the function and documentation


def run(images_path="media/", filename="", num_of_results = 1, hm_lvl = 0, certainty=0, data_dir=DATA_DIR):
    """execute face detection than vgg face and finally grad-cam

    :param filename: query image filename (default = "")
    :param images_path: query and output image path (default = "media/")
    :param data_dir: openCV directory path (default = DATA_DIR)
    :return TODO
    """

    res = Results(certainty)

    file_path = os.path.join(images_path, filename)
    if not file_path:
        res.err_msg = "ERROR: cannot load input image {}".format(filename)
        res.err_code = 1
        message(res.err_msg)
        return res

    face = FaceDetect(data_dir, file_path)
    if not face.is_valid:
        res.err_msg = "ERROR: cannot load input image {}".format(filename)
        res.err_code = 1
        message(res.err_msg)
        return res

    face.load_cascades()
    if not face.is_loaded:
        res.err_msg = "ERROR: cannot load cascades from: {}".format(data_dir)
        res.err_code = 2
        message(res.err_msg)
        return res

    face.detect_face()
    if not face.has_face:
        res.err_msg = "ERROR: sorry, frontal face wasn't detected"
        res.err_code = 3
        message(res.err_msg)
        return res

    # Crop faces from query image
    im = cv2.imread(file_path)
    # If found more than one face, pick the biggest one (w * h)
    cropped_im = crop_rect(im, max(face.features, key=lambda f: f[2] * f[3]))
    cv2.imwrite(os.path.join(images_path, "cropped.jpg"), cropped_im)

    # Run forward pass and GradCam on cropped image
    pred_labels, err_msg = classifier_gcam.predict(images_path, num_of_results)
    if (pred_labels is None) or (len(pred_labels) == 0):
        res.err_msg = err_msg
        res.err_code = 4
        message(res.err_msg)
        return res

    # Get predicted label from Torch output
    pred_ids = get_prediction_from_names(pred_labels)
    if len(pred_labels) == 0:
        res.err_msg = "ERROR: could not load names.txt file"
        res.err_code = 5
        message(res.err_msg)
        return res

    res.set_results(cropped_im, pred_ids[0], pred_labels[0], pred_ids[1:], hm_lvl)
    res.find_significant_features()

    return res
