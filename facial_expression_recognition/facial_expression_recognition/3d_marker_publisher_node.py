# Author : Apurv Saha
# Description : This script is used to detect facial landmarks and draw lines around the face regions and display using the Foxglove library.
# It uses ROS2 and Python3.8.5.

import cv2
import rclpy
import numpy as np
from rclpy.node import Node
import dlib
from imutils.face_utils import FACIAL_LANDMARKS_68_IDXS
from cv_bridge import CvBridge
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


from keras import models

from std_msgs.msg import ColorRGBA, String
from sensor_msgs.msg import Image
from foxglove_msgs.msg import ImageMarkerArray
from visualization_msgs.msg import ImageMarker
from geometry_msgs.msg import Point

cv_bridge = CvBridge()
face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1("/vagrant/src/3d_labels_facial_expressions_foxglove/mmod_human_face_detector.dat")
resnet_face_detector = dlib.face_recognition_model_v1("/vagrant/src/3d_labels_facial_expressions_foxglove/dlib_face_recognition_resnet_model_v1.dat")
predictor = dlib.shape_predictor("/vagrant/src/3d_labels_facial_expressions_foxglove/shape_predictor_68_face_landmarks.dat")

# Deep learning based face derector
protoxt = "/vagrant/src/3d_labels_facial_expressions_foxglove/deploy.prototxt"
model = "/vagrant/src/3d_labels_facial_expressions_foxglove/res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoxt, model)

# ROS ColorRGBA messages for each facial region
COLORS = [
    ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),
    ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
    ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
    ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
    ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
]

# OpenCV colors for each facial region
COLORS_BGR = [
    (0, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
]

Emotion_COLOR = {
    'Angry': (0, 0, 255),
    'Fear': (255, 0, 255),
    'Disgust': (0, 255, 0),
    'Happy': (0, 255, 255),
    'Neutral': (255, 255, 255),
    'Sad': (0, 0, 255),
    'Surprise': (0, 0, 255),
}


# Emotion Detection
emotion_predictor = load_model("/vagrant/src/3d_labels_facial_expressions_foxglove/best_model.h5")

class_labels = ['Angry', 'Fear', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Surprise']

class FacialMarkerPublisherNode(Node):
    """
    This class is used to detect facial landmarks and draw lines around the face regions and display using the Foxglove library.
    """

    def __init__(self):
        super().__init__("facial_marker_publisher_node")
        self.pub_markers = self.create_publisher(
            ImageMarkerArray, "/usb_cam/face_markers", 1
        )
        self.pub_image = self.create_publisher(
            Image, "/usb_cam/image_raw", 1
        )
        self.create_subscription(
            Image, "/image/cam2", self.image_callback, 1
        )
        self.pub_emotion = self.create_publisher(
            String, "/facial_expression", 1
        )
            

    def image_callback(self, msg: Image):
        # Convert the ROS Image to a grayscale OpenCV image
        cv_img = cv_bridge.imgmsg_to_cv2(msg)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        grayscale_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        image_height, image_width, _ = rgb_img.shape

        # Run the face detector on the grayscale image
        rects = face_detector(rgb_img, 0)
      

        dlib_rectangles = dlib.rectangles()

        processed_image = cv2.dnn.blobFromImage(cv_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(processed_image)
        detections = net.forward()
        self.get_logger().info(f"Detected {len(detections)} faces using DNN")

        markers = ImageMarkerArray()
        for face in detections[0, 0]:
            face_confidence = face[2]
            if face_confidence > 0.5:
                bbox = face[3:7] * np.array([image_width, image_height, image_width, image_height])

                self.get_logger().info(f"Detected face: {bbox}")

                rect_dlib = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                # Run the predictor, which returns a list of 68 facial landmarks as (x,y) points
                points = predictor(rgb_img, rect_dlib).parts()

                # self.get_logger().info(f"Detected {len(points)} facial landmarks")

                # Get ROI for the face
                face_roi = rgb_img[
                    int(bbox[1]):int(bbox[3]),
                    int(bbox[0]):int(bbox[2])
                ]
                face_roi = cv2.resize(face_roi, (224, 224))
                image_pixels = img_to_array(face_roi)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255.0

                # Predict the emotion
                predictions = emotion_predictor.predict(image_pixels)
                self.get_logger().info(f"Predictions: {predictions}")
                emotion_label = class_labels[np.argmin(predictions[0])]



                # Draw the face bounding box and write the emotion label
                cv2.rectangle(
                    rgb_img,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    Emotion_COLOR[class_labels[np.argmin(predictions[0])]],
                    2,
                )
                cv2.putText(
                    rgb_img,
                    emotion_label,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Emotion_COLOR[class_labels[np.argmin(predictions[0])]],
                    2,
                )

                # Publish the emotion label
                emotion_ros_msg = String()
                emotion_ros_msg.data = emotion_label
                self.pub_emotion.publish(emotion_ros_msg)


                # Draw a line around each face region
                for region_idx, (name, (start_idx, end_idx)) in enumerate(
                    FACIAL_LANDMARKS_68_IDXS.items()
                ):
                    region_points = points[start_idx:end_idx]
                    # Connect the points for each region in a loop, except for the jaw
                    if name != "jaw":
                        region_points.append(region_points[0])

                    # Draw the lines on the image
                    for i in range(1, len(region_points)):
                        cv2.line(
                            rgb_img,
                            (region_points[i - 1].x, region_points[i - 1].y),
                            (region_points[i].x, region_points[i].y),
                            COLORS_BGR[region_idx % len(COLORS)],
                            1,
                        )

                    markers.markers.append(
                        ImageMarker(
                            header=msg.header,
                            scale=1.0,
                            type=ImageMarker.LINE_STRIP,
                            points=[Point(x=float(p.x), y=float(p.y)) for p in region_points],
                            outline_color=COLORS[region_idx % len(COLORS)],
                        )
                    )
        
        # Publish the markers
        self.pub_markers.publish(markers)
        # Publish the image
        self.pub_image.publish(cv_bridge.cv2_to_imgmsg(rgb_img))


def main(args=None):
    rclpy.init(args=args)
    node = FacialMarkerPublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
