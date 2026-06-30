import cv2

class Camera:
    def __init__(self, url):
        # Stream öffnen (OpenCV übernimmt das Decoding)
        self.stream = cv2.VideoCapture(url)

    def get_frame(self):
        ok, frame = self.stream.read()
        if not ok or frame is None:
            return None

        # WICHTIG: Bild verkleinern, damit OpenCV den Crop/Zoom-Bug umgeht
        # Du kannst die Größe anpassen, aber 640x480 funktioniert fast immer perfekt
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        return frame
