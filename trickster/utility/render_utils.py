import cv2


class Mode:

    HUMAN = "human"
    FILE = "file"


class FileRenderer:

    def __init__(self, output_path: str, fps=25):
        self.output_path = output_path
        self.fps = fps
        self.size = None
        self.device = None
        self._in_ctx = False

    def append(self, frame):
        if not self._in_ctx:
            raise RuntimeError("Must be called in a context manager!")
        if self.device is None:
            self._open(frame.shape)
        self.device.write(frame)

    def _open(self, shape: list):
        shape = shape[:2][::-1] + shape[-1]
        fcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.device = cv2.VideoWriter(self.output_path, fcc, self.fps, shape)

    def __enter__(self):
        self._in_ctx = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_ctx = False
        self.device.close()
