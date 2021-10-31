import cv2
import numpy as np
from tqdm import tqdm


CONF_THRESHOLD = 0.5
SUPPRESSION_THRESHOLD = 0.3
YOLO_IMG_SIZE = 320
PLANE_CLASS = 4
MIN_WIDTH = 108     # 128 - po 10 px z każdej strony


class PlaneBB():
    def __init__(self, x, y, w, h, image, conf=None):
        im_height, im_width = image.shape[:2]
        self.image = image
        self.x1 = x
        self.x1 = max(self.x1 - 10, 0)          # by nie wylazło poza obraz
        self.y1 = y
        self.x2 = x + w
        self.x2 = min(self.x2 + 10, im_width)   # by nie wylazło poza obraz
        self.y2 = y + h

        self.h_new = w * (9/16)                 # tworzymy nową wysokość obrazu
        if self.h_new > h:
            self.y1 -= int((self.h_new - h) / 2)
            self.y2 += int((self.h_new - h) / 2)

        self.y1 = max(self.y1 - 10, 0)          # by nie wylazło poza obraz
        self.y2 = min(self.y2 + 10, im_height)  # by nie wylazło poza obraz
        self.conf = conf

    def save_plane_img(self, path):
        roi = self.image[self.y1: self.y2, self.x1: self.x2]
        cv2.imwrite(path, roi)


def find_planes(model_outputs, im_height, im_width):
    bbox_locations = []
    confidences = []

    for output in model_outputs:
        # consider every prediction in layer
        for prediction in output:
            # wszystkie oprócz pierwszych 5 parametrów to prawdopodobieństwa klas
            class_probabilities = prediction[5:]
            most_prob_class = np.argmax(class_probabilities)
            confidence = class_probabilities[most_prob_class]

            if confidence > CONF_THRESHOLD and most_prob_class == PLANE_CLASS:
                x = int((prediction[0] - prediction[2]/2) * im_width)
                x = max(0, x)   # na przypadek, gdyby znajdywał ujemne liczby
                y = int((prediction[1] - prediction[3]/2) * im_height)
                y = max(0, y)   # na przypadek, gdyby znajdywał ujemne liczby
                w = int(prediction[2] * im_width)
                h = int(prediction[3] * im_height)
                if w >= MIN_WIDTH:
                    bbox_locations.append([x, y, w, h])
                    confidences.append(float(confidence))  # musi być we float, inaczej dziwny błąd :p

    # non-maximum suppression
    bbox_indices = cv2.dnn.NMSBoxes(bboxes=bbox_locations, scores=confidences,
                                    score_threshold=CONF_THRESHOLD, nms_threshold=SUPPRESSION_THRESHOLD)
    bbox_locations = [bbox_locations[idx[0]] for idx in bbox_indices]

    return bbox_locations


def get_blob(img):
    return cv2.dnn.blobFromImage(img, scalefactor=1/255,
                                 size=(YOLO_IMG_SIZE, YOLO_IMG_SIZE), swapRB=True, crop=False)


def initialize_network():
    network = cv2.dnn.readNetFromDarknet(cfgFile='yolov3.cfg', darknetModel='yolov3.weights')

    # set device (CPU or GPU)
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return network


def video_loading(video_path, interval):
    frame_nr = 0
    frames = []
    video = cv2.VideoCapture(video_path)
    print("Wczytuję film...")

    while video.isOpened():
        is_grabbed, frame = video.read()

        # koniec filmu
        if not is_grabbed:
            break

        if frame_nr % interval == 0:
            print(f'Wczytuję klatkę {int(frame_nr/interval)}')
            frames.append(frame)
        frame_nr += 1

    video.release()
    cv2.destroyAllWindows()
    return frames


def network_pass(image):
    # defenujemy blob
    blob = get_blob(image)
    # karmimy sieć blobem
    network.setInput(blob)
    # znajdujemy warstwy, które dokonują predykcji
    output_layers = network.getUnconnectedOutLayersNames()
    # forward pass
    outputs = network.forward(output_layers)

    return outputs


#variables
frame_check_interval = 100
video_name = 'B787.mp4'
video_path = f'videos/{video_name}'
img_name = video_name.split('.')[:-1][0]

network = initialize_network()
frames = video_loading(video_path, frame_check_interval)

# określame rozmiary wideo
im_height, im_width = frames[0].shape[:2]


# main loop
for frame_nr, frame in tqdm(enumerate(frames)):
    # przepuszczamy klatki przez sieć
    outputs = network_pass(frame)
    # znajdujemy objekty YOLO na warstwach wyjściowych sieci
    bbox_locations = find_planes(outputs, im_height=im_height, im_width=im_width)
    # tworzymy objekty bounding boksów i zapisujemy obrazy
    for obj_nr, bbox in enumerate(bbox_locations):
        plane_bb = PlaneBB(bbox[0], bbox[1], bbox[2], bbox[3], frame)
        plane_bb.save_plane_img(f'dataset/{img_name}_{frame_nr}_{obj_nr}.jpg')
