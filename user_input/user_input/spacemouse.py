import pyspacemouse
import time
import threading


class SpaceMouse:
    """
    Reading from a 3D Connexion SpaceMouse device.
    """

    def __init__(self):
        pyspacemouse.open()
        self._running = False
        self._thread_lock = threading.Lock()
        self._state = pyspacemouse.read()

    def __enter__(self):
        if not self._running:
            self.start_listener()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._running:
            self.stop_listener()

    def start_listener(self):
        self._thread = threading.Thread(
            target=self._listener, args=[], kwargs={}
        )
        self._running = True
        self._thread.start()

    def stop_listener(self):
        self._running = False
        self._thread.join()

    def _listener(self):
        """
        Worker thread conintiously polling the space mouse to get current state.
        """
        # more accurate control rate when compared with calling time.sleep()
        control_rate = 1.0 / 100.0
        _start_time = time.perf_counter()

        while self._running:
            _next_time = _start_time + control_rate

            with self._thread_lock:
                self._state = pyspacemouse.read()

            # enforce accurate control rate
            while time.perf_counter() < _next_time:
                pass

            _start_time = _next_time

    def get_state(self):
        with self._thread_lock:
            state = self._state
        return state


if __name__ == '__main__':

    spacemouse = SpaceMouse()
    with spacemouse:
        for i in range(1000):
            print(spacemouse.get_state())
            time.sleep(0.01)
