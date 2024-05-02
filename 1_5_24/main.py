import cv2
import numpy as np
import threading
import time

from class_bien_bao import ObjectDetector

from class_lane_keeping import LaneDetectionThread

def sign_detection_thread(video_path):
    cap = cv2.VideoCapture(video_path)
    thread_object = ObjectDetector(cap)
    thread_object.start()
    thread_object.join()


def lane_detection_thread(video_path):
    cap = cv2.VideoCapture(video_path)
    thread_lane = LaneDetectionThread(cap)
    thread_lane.start()
    thread_lane.join()

if __name__ == "__main__":
    cap1 = cv2.VideoCapture(0)
    video_path="Linnk vid"  

    # Create instances of MyThread1 and CameraCaptureThread
    thread1 = threading.Thread(target=sign_detection_thread, args=(video_path,))
    thread3 = threading.Thread(target=lane_detection_thread, args=(video_path,))

    # Start both threads
    thread1.start()
    thread3.start()


    # Wait for both threads to finish
    thread1.join()
    thread3.join()


    print("All threads have finished")
