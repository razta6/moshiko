import numpy as np
import cv2
import os
from support import DATA_DIR, NET_HEIGHT, NET_WEIGHT, message, debug, warning, fatal

OPENCV_PATH = DATA_DIR + 'haarcascades/'
NUM_CHANNELS = 3
GREY_BLEND = 125
FEATURES_CASCADES = {
    "eyes": "haarcascade_eye.xml",
    "mouth": "haarcascade_mcs_mouth.xml",
    "nose": "haarcascade_mcs_nose.xml",
    "eyebrows": "",
    "forehead": "",
}


class Results:
    def __init__(self, certainty, threshold=GREY_BLEND):
        """
        Creates an empty "Results" object.
        :param threshold: The minimum level (0-255) per channel to consider significant.
        """
        self.im = []
        self.heat_map = []
        self.top1_id = ""
        self.top1_label = -1
        self.top_other_ids = []
        self.err_code = 0
        self.err_msg = ""
        self.output_filename = ""
        self.s_features = []
        self.threshold = threshold + certainty*15
        # 0 - RGB, 1 = RG, 2 = R
        self.hm_lvl = 0

    def set_results(self, im, pred_id, pred_label, all_preds, hm_lvl):
        """
        Populates the "Results" object.
        These atributes will be used for the results display to the user.
        :param im: User cropped image (ndarray)
        :param pred_id: Top predicted id (name) by the network
        :param pred_label: Top predicted label (number) by the network
        :param all_preds: Other predictions (based on number of results requested)
        :param hm_lvl: Filter level to apply on the heat map
        """
        self.im = cv2.resize(im, (NET_WEIGHT, NET_HEIGHT))
        self.top1_id = pred_id
        self.top1_label = pred_label
        self.top_other_ids = all_preds
        self.hm_lvl = hm_lvl
        debug("Results set.")

    def _load_heat_map(self):
        """
        Loads the heat map that was generated from the grad-CAM into self.heat_map
        """
        heat_map_path = "media/heat_map_{}.png".format(self.top1_label)
        heat_map = cv2.imread(heat_map_path)
        self.heat_map = cv2.resize(heat_map, (NET_WEIGHT, NET_HEIGHT))
        debug("Heat map loaded.")

    def _find_feature(self, feature, feature_cascade):
        """
        Finds the requested feature by its cascade (if exists).
        :param feature: Name of feature to find
        :param feature_cascade: Cascade filename for the requested feature (if exists).
        :return:
        """
        # For eyebrows and forehead
        if feature_cascade == "":
            return []

        feature_cascade = cv2.CascadeClassifier(os.path.join(OPENCV_PATH, feature_cascade))
        color = cv2.cvtColor(self.im, cv2.CV_8U)
        # Params are changeable, 1.05 (bigger is stricter), 3 (bigger is stricter)
        features = feature_cascade.detectMultiScale(color, 1.1, 2)
        if len(features) == 0 or (feature == 'eyes' and len(features) < 2):
            debug("Couldn't find feature - {}".format(feature))
            return []

        # Find both eyes
        if feature == 'eyes':
            eye_center_x = lambda coords: (coords[0] + coords[2]) / 2
            # Sort by eye center (x axis)
            eyes = sorted(features, key=eye_center_x)
            ret = [eyes[0], eyes[-1]]  # Left eye, right eye
        else:
            ret = [features[0]]  # [(x, y, w, h)]

        return ret

    def _find_eyebrows(self, eyes_coords):
        """
        Eyebrows do not have a cascade, so we find them using edge detection.
        :param eyes_coords: A list with rectangle coordinates for both eyes found [left_eye, right_eye].
        :return: A list with eyebrows coordinates found, or empty if found less than 2.
        """
        ret = []
        for x_e, y_e, w_e, h_e in eyes_coords:
            edges = cv2.Canny(self.im[y_e: y_e + h_e, x_e: x_e + w_e], 50, 200)
            edges_ones = edges.nonzero()
            # Check for a minimum of existing edges
            if len(edges_ones) < 2 or len(edges_ones[0]) < 10 or len(edges_ones[1]) < 5:
                debug("Not enough edges found for eyebrows.")
                return []
            # Found enough edges
            x_eb, w_eb = x_e, w_e  # Same as eye coords
            # Find top of eyebrow
            y_eb = min(edges_ones[0])
            # Find bottom of eyebrow
            min_index = np.argmin(edges_ones[0])
            mid_x = edges_ones[1][min_index]
            curr_x = mid_x
            curr_y = y_eb

            is_in_bounds = lambda x, y: x >= 0 and y >= 0 and x < len(edges[0]) and y < len(edges)

            # Find left bottom
            while is_in_bounds(curr_x, curr_y) and edges[curr_y][curr_x]:
                # Try left down
                if is_in_bounds(curr_x - 1, curr_y + 1) and edges[curr_y + 1][curr_x - 1]:
                    curr_y += 1
                    curr_x -= 1
                    continue
                # Try only down
                if is_in_bounds(curr_x, curr_y + 1) and edges[curr_y + 1][curr_x]:
                    curr_y += 1
                    continue

                # Try only left
                if is_in_bounds(curr_x - 1, curr_y) and edges[curr_y][curr_x - 1]:
                    curr_x -= 1
                    continue
                # Can't go further, give some slack
                curr_x -= 1
                curr_y += 1
            h_l = curr_y - y_eb

            # Find right bottom
            curr_x = mid_x
            curr_y = y_eb
            while is_in_bounds(curr_x, curr_y) and edges[curr_y][curr_x]:
                # Try right down
                if is_in_bounds(curr_x + 1, curr_y + 1) and edges[curr_y + 1][curr_x + 1]:
                    curr_y += 1
                    curr_x += 1
                    continue
                # Try only down
                if is_in_bounds(curr_x, curr_y + 1) and edges[curr_y + 1][curr_x]:
                    curr_y += 1
                    continue

                # Try only right
                if is_in_bounds(curr_x + 1, curr_y) and edges[curr_y][curr_x + 1]:
                    curr_x += 1
                    continue
                # Can't go further, give some slack
                curr_x += 1
                curr_y += 1
            h_r = curr_y - y_eb

            # Find minimum of eyebrow tips
            h_eb = max(h_r, h_l)
            # y_eb was in eye section coordinates
            y_eb += y_e
            ret += [(x_eb, y_eb, w_eb, h_eb)]

        return ret

    def _get_feature_score(self, coords):
        """
        Returns a sum of weighted average (per channel) of the feature area on the heat map.
        If a channel is to be filtered, based on hm_lvl, it is not calculated in the score.
        The higher the score, the more significant the feature is.
        :param coords: Feature rectangle coordinates.
        :return: A list of weighted averages per channel (non-filtered ones) [..., avg(G), avg(R)]
        """
        # Weighted average pixel of area detected, by channel
        # B, G, R channels are multiplied by 1, 2, 3 respectively
        # Results in: [[avg(B1), avg(G1), avg(R1)], ... , [avg(Bn), avg(Gn), avg(Rn)]]
        wavgs = [[np.average(self.heat_map[y: y + h, x: x + w, c]) * (c + 1) for c in range(self.hm_lvl, NUM_CHANNELS)]
                 for x, y, w, h in coords]
        debug("Weighted channel averages before summation: {}".format(wavgs))
        # Other options to consider:
        # Not weighted average:
        # avgs = [[np.average(self.heat_map[y: y + h, x: x + w, c]) for c in range(self.hm_lvl, NUM_CHANNELS)] for x, y, w, h in coords]
        # BGR channels weighted by powers of 1, 2, 3 respectively
        # pavgs = [[np.average(self.heat_map[y: y + h, x: x + w, c]) * ((c + 1) ** 2) for c in range(self.hm_lvl, NUM_CHANNELS)] for x, y, w, h in coords]
        return np.average(wavgs, axis=0)

    def find_significant_features(self):
        """
        Finds features in the face, gets their scores and filters the least significant ones.
        In the end, self.s_features will have a list of significant feature names, ordered from high to low.
        """
        # Load the heat map and blend with user image
        self._load_heat_map()
        self._create_blended()
        
        # Find signiicant features
        feature_coords = dict()
        for feature in FEATURES_CASCADES:
            feature_coords[feature] = self._find_feature(feature, FEATURES_CASCADES[feature])

        if not len(feature_coords):
            self.err_code = 6
            self.err_msg = "No significant features found."
            warning(self.err_msg)

        # Find eyebrows and forehead if eyes were found
        if len(feature_coords['eyes']) == 2:
            eyebrows = self._find_eyebrows(feature_coords['eyes'])
            # If found, find forehead and adjust eyes
            if len(eyebrows) == 2:
                x_eb_l, y_eb_l, w_eb_l, h_eb_l = eyebrows[0]
                x_eb_r, y_eb_r, w_eb_r, h_eb_r = eyebrows[1]
                y_eb = min(y_eb_l, y_eb_r)
                x_eb = x_eb_l
                w_eb = x_eb_r + w_eb_r - x_eb_l
                # Forehead
                feature_coords['forehead'] = [(x_eb, 0, w_eb, y_eb)]
                # Subtract half of eyebrows height from eyes
                x_e_l, y_e_l, w_e_l, h_e_l = feature_coords['eyes'][0]
                x_e_r, y_e_r, w_e_r, h_e_r = feature_coords['eyes'][1]
                h_e_l = y_e_l + h_e_l - int(y_eb_l + h_eb_l / 2)
                h_e_r = y_e_r + h_e_r - int(y_eb_r + h_eb_r / 2)
                y_e_l = int(y_eb_l + h_eb_l / 2)
                y_e_r = int(y_eb_r + h_eb_r / 2)
                feature_coords['eyes'] = [(x_e_l, y_e_l, w_e_l, h_e_l), (x_e_r, y_e_r, w_e_r, h_e_r)]
                feature_coords['eyebrows'] = eyebrows

        for feature in feature_coords:
            coords = feature_coords[feature]
            # Feature not found
            if not len(coords):
                continue
            debug("Feature coords: {}".format(coords))
            self.s_features.append((feature, self._get_feature_score(coords)))
        # Sort features from highest to lowest
        self.s_features.sort(reverse=True, key=lambda f: f[1][-1])
        debug("Found features before filtering: {}".format(self.s_features))
        # Filter out un-distinct features and leave only names
        self.s_features = [feature[0] for feature in self.s_features if self._is_distinct_feature(feature[1])]
        
    def _is_distinct_feature(self, feature_score):
        """
        Filter undistinct features by threshold, by filter in descending order.
        :param feature_score: A list of weighted averages per non-filtered channels
        :return: True if feature is distinct, False otherwise
        """
        for channel in range(NUM_CHANNELS - 1, self.hm_lvl - 1, -1):
            num_weights = np.sum(range(NUM_CHANNELS, channel, -1))
            # Accounted for channels
            score = np.sum(feature_score[channel - self.hm_lvl:])
            if score >= self.threshold * num_weights:
                return True
        return False

    def _create_blended(self):
        """
        Blends the heat map (filtered based on hm_lvl) with the use image.
        Saves it in the "media" folder.
        """
        hm_cpy = self.heat_map.copy()
        # Make filtered channels to be neutral after blend
        for c in range(self.hm_lvl):
            hm_cpy[:, :, c] = GREY_BLEND
        blended = cv2.addWeighted(self.im, 0.5, hm_cpy, 0.5, 0)
        self.output_filename = "blended_{}.jpg".format(self.top1_label)
        cv2.imwrite(os.path.join("media/", self.output_filename), blended)
