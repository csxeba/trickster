from typing import List

import cv2


class Mode:

    SCREEN = "screen"  # render to the screen
    FILE = "file"  # render to a video file


class Renderer:

    def __init__(self, fps=25, scaling_factor: float = 1.):
        self.fps = fps
        self.size = None
        self.device = None
        self.scale = scaling_factor
        self._in_ctx = False

    def append(self, frame):
        if not self._in_ctx:
            raise RuntimeError("Must be called in a context manager!")
        if self.device is None:
            self._open(frame.shape)
        if self.scale != 1.:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        self._write(frame)

    def _open(self, shape):
        raise NotImplementedError

    def _write(self, frame):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError

    def __enter__(self):
        self._in_ctx = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_ctx = False
        self._close()


class FileRenderer(Renderer):

    def __init__(self, output_path: str, fps: int = 25, scaling_factor: float = 1.):
        super().__init__(fps, scaling_factor)
        self.output_path = output_path

    def _open(self, shape: tuple):
        shape = shape[:2][::-1] + (shape[-1],)
        fcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.device = cv2.VideoWriter(self.output_path, fcc, self.fps, shape)

    def _write(self, frame):
        self.device.write(frame)

    def _close(self):
        self.device.release()


class ScreenRenderer(Renderer):

    def __init__(self, window_name: str = "screen", fps: int = 25, scaling_factor: float = 1.):
        super().__init__(fps, scaling_factor)
        self.window_name = window_name

    def _open(self, shape):
        self.device = cv2.namedWindow(self.window_name)

    def _write(self, frame):
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1000 // self.fps)

    def _close(self):
        cv2.destroyWindow(self.window_name)


class MultiRenderer(Renderer):

    def __init__(self, renderers: List[Renderer], fps: int = 25, scaling_factor: float = 1.):
        super().__init__(fps, scaling_factor)
        self.renderers = renderers
        for renderer in self.renderers:
            renderer.scale = 1.

    def _open(self, shape):
        for renderer in self.renderers:
            renderer._open(shape)

    def _write(self, frame):
        for renderer in self.renderers:
            renderer._write(frame)

    def _close(self):
        for renderer in self.renderers:
            renderer._close()


def factory(screen_name: str = None,
            output_file_path: str = None,
            fps: int = 25,
            scaling_factor: float = 1.) -> Renderer:

    renderers = []

    if screen_name is not None:
        renderers.append(ScreenRenderer(screen_name, fps, scaling_factor))
    if output_file_path is not None:
        renderers.append(FileRenderer(output_file_path, fps))

    if len(renderers) == 1:
        return renderers[0]
    return MultiRenderer(renderers)
