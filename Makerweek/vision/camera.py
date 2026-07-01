import cv2
import time
import threading


class Camera:
    def __init__(self, url):
        self._url       = url
        self._frame     = None
        self._timestamp = 0.0
        self._lock      = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        stream = cv2.VideoCapture(self._url)
        while True:
            ok, frame = stream.read()
            if ok and frame is not None:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._frame     = frame
                    self._timestamp = time.time()
            else:
                print("Kamera-Verbindung unterbrochen – reconnecte...")
                stream.release()
                time.sleep(0.5)
                stream = cv2.VideoCapture(self._url)

    def get_frame(self):
        """Gibt (frame, timestamp) zurück. timestamp = Zeitpunkt der Aufnahme."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._timestamp
