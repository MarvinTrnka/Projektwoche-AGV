import cv2

def show_debug(img, grid, pose, path):
    # Zeichne Marker, Grid, Pfad, Pose
    cv2.imshow("debug", img)
    cv2.waitKey(1)