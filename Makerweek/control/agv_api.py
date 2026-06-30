import requests

class AGV:
    def __init__(self, ip):
        self.base = f"http://{ip}"

    def drive(self, left_steps, right_steps):
        requests.post(self.base + "/drive", json={
            "left": left_steps,
            "right": right_steps
        })