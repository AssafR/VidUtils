import cv2


class WindowManager:
    def __init__(self, window_name='Frame Window'):
        self.window_name = window_name

    def __enter__(self):
        cv2.namedWindow(self.window_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        while cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) > 0:
            if cv2.waitKey(1) != -1:
                break
        cv2.destroyAllWindows()


class WindowClosed(Exception):
    pass