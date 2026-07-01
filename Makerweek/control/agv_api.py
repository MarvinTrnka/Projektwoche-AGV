import requests
import time


class AGV:
    def __init__(self, ip):
        self.base = f"http://{ip}"

    def enable(self):
        """Schaltet die Stepper-Motoren ein."""
        try:
            r = requests.post(self.base + "/api/agv/stepper/enable",
                              json={"stepper": "on"}, timeout=3)
            print(f"enable → {r.status_code}  {r.text[:120]}")
        except Exception as e:
            print(f"enable FEHLER: {e}")

    def disable(self):
        try:
            r = requests.post(self.base + "/api/agv/stepper/enable",
                              json={"stepper": "off"}, timeout=3)
            print(f"disable → {r.status_code}")
        except Exception as e:
            print(f"disable FEHLER: {e}")

    def abort(self):
        """Stoppt alle Motorbewegungen sofort."""
        try:
            requests.post(self.base + "/api/agv/stepper/abortMotion", timeout=3)
        except Exception as e:
            print(f"abort FEHLER: {e}")

    def set_max_velocity(self, percent=100):
        """Setzt maximale Geschwindigkeit (0–100 %)."""
        try:
            r = requests.post(self.base + "/api/agv/stepper/setMaxVelocityPerc",
                              json={"maxVel_perc": float(percent)}, timeout=3)
            print(f"set_velocity({percent}%) → {r.status_code}  {r.text[:80]}")
        except Exception as e:
            print(f"set_velocity FEHLER: {e}")

    def set_max_acceleration(self, percent=50):
        """Setzt maximale Beschleunigung (0–100 %)."""
        try:
            r = requests.post(self.base + "/api/agv/stepper/setMaxAccelerationPerc",
                              json={"maxAcc_perc": float(percent)}, timeout=3)
            print(f"set_acceleration({percent}%) → {r.status_code}  {r.text[:80]}")
        except Exception as e:
            print(f"set_acceleration FEHLER: {e}")

    def drive(self, left_steps, right_steps):
        """Fährt relativ: positive Werte = vorwärts."""
        try:
            r = requests.post(self.base + "/api/agv/stepper/setMoveRelative",
                              json={"leftDelta_steps":  left_steps,
                                    "rightDelta_steps": right_steps},
                              timeout=3)
            print(f"drive(L={left_steps}, R={right_steps}) → {r.status_code}  {r.text[:120]}")
        except Exception as e:
            print(f"drive FEHLER: {e}")

    def is_moving(self):
        try:
            r = requests.get(self.base + "/api/agv/stepper/isMoving", timeout=1)
            return r.json().get("data", {}).get("isMoving", False)
        except Exception:
            return False

    def wait_for_stop(self, timeout=15.0):
        """Blockiert bis Bewegung fertig (max. timeout s)."""
        deadline = time.time() + timeout
        time.sleep(0.1)
        while time.time() < deadline:
            if not self.is_moving():
                return
            time.sleep(0.05)
