import cv2
import math
import os
from support import PROFILES


class FaceDetect:
    def __init__(self, data_dir, file_path):
        self.data_dir = data_dir
        self.file_path = file_path
        self.im = None
        self.features = []
        self.cascades = {}
        self.has_face = False
        self.is_loaded = False

        self.is_valid = True
        self.im = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        if self.im is None:
            self.is_valid = False

    def load_cascades(self):
        """ Loads cascades into Cascades array
        self.is_loaded = True on success, False otherwise
        """
        if not self.is_valid:
            self.is_loaded = False
            return
        for k, v in PROFILES.items():
            v = os.path.join(self.data_dir, v)
            try:
                if not os.path.exists(v):
                    raise cv2.error('no such file')
                self.cascades[k] = cv2.CascadeClassifier(v)
            except cv2.error:
                self.is_loaded = False
        self.is_loaded = True

    def detect_face(self):
        self.im = cv2.equalizeHist(self.im)
        side = math.sqrt(self.im.size)
        minlen = int(side / 20)
        maxlen = int(side / 2)
        flags = cv2.CASCADE_DO_CANNY_PRUNING

        cc = self.cascades['HAAR_FRONTALFACE_DEFAULT']
        self.features = cc.detectMultiScale(self.im, 1.1, 4, flags, (minlen, minlen), (maxlen, maxlen))
        if not len(self.features):
            cc = self.cascades['HAAR_FRONTALFACE_ALT']
            self.features = cc.detectMultiScale(self.im, 1.1, 4, flags, (minlen, minlen), (maxlen, maxlen))
        if not len(self.features):
            cc = self.cascades['HAAR_FRONTALFACE_ALT2']
            self.features = cc.detectMultiScale(self.im, 1.1, 4, flags, (minlen, minlen), (maxlen, maxlen))
        if not len(self.features):
            cc = self.cascades['HAAR_PROFILEFACE']
            self.features = cc.detectMultiScale(self.im, 1.1, 4, flags, (minlen, minlen), (maxlen, maxlen))
        if len(self.features):
            self.has_face = True
