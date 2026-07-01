"""
TestDrive.py — Motorkalibrierung
=================================
Befehle im Terminal eingeben + Enter drücken.

  f  — vorwärts (STEPS_TEST Steps)
  b  — rückwärts
  l  — links  90°   (STEPS_TURN Steps)
  r  — rechts 90°   (STEPS_TURN Steps)
  h  — links 180°   (2 × STEPS_TURN) ← am besten für genaue Kalibrierung!
  +  — STEPS_TURN um 5 erhöhen (Drehung größer)
  -  — STEPS_TURN um 5 verringern (Drehung kleiner)
  ]  — STEPS_TEST um 10 erhöhen (Fahrstrecke größer)
  [  — STEPS_TEST um 10 verringern (Fahrstrecke kleiner)
  v  — Geschwindigkeit um 5% erhöhen
  s  — Geschwindigkeit um 5% verringern
  q  — beenden + Werte ausgeben

Kalibrierungs-Tipp:
  'h' für 180°: AGV dreht und sollte exakt in die entgegengesetzte Richtung zeigen.
  Fehler bei 180° sind doppelt so gut sichtbar wie bei 90°.
  Dann +/- anpassen bis 'h' perfekt ist → 'l'/'r' stimmt automatisch auch.

Ziel:
  STEPS_TEST → 1 Gitterzelle (~5 cm)  → in kinematics.py als STEPS_PER_CELL
  STEPS_TURN → genau 90°              → STEPS_PER_RAD = STEPS_TURN / (pi/2)
"""

import math
import time
from control.agv_api import AGV

AGV_IP     = "172.17.1.47"
STEPS_TEST = 200    # Steps vorwärts/rückwärts
STEPS_TURN = 130    # Steps für 90° Drehung — Startwert basierend auf Kalibrierung
VEL        = 20     # Geschwindigkeit %

agv = AGV(AGV_IP)

print(f"Verbinde mit AGV @ {AGV_IP} ...")
agv.enable()
agv.set_max_velocity(VEL)
agv.set_max_acceleration(20)   # Niedrig = sanft, kein Schritt-Überspringen
print(f"Bereit.  VEL={VEL}%  STEPS_TEST={STEPS_TEST}  STEPS_TURN={STEPS_TURN}")
print("f=vor  b=zurück  l=links90°  r=rechts90°  h=links180°  +/-=Drehung±5  ]/[=Strecke±10  v/s=Tempo  q=Ende")

while True:
    try:
        cmd = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        break

    if cmd == 'q':
        break

    elif cmd == 'f':
        print(f"Vorwärts {STEPS_TEST} Steps...")
        agv.drive(STEPS_TEST, STEPS_TEST)
        agv.wait_for_stop(timeout=10)
        print("Fertig.")

    elif cmd == 'b':
        print(f"Rückwärts {STEPS_TEST} Steps...")
        agv.drive(-STEPS_TEST, -STEPS_TEST)
        agv.wait_for_stop(timeout=10)
        print("Fertig.")

    elif cmd == 'l':
        print(f"Links drehen {STEPS_TURN} Steps...")
        agv.drive(-STEPS_TURN, STEPS_TURN)
        agv.wait_for_stop(timeout=10)
        print("Fertig.")

    elif cmd == 'r':
        print(f"Rechts drehen {STEPS_TURN} Steps...")
        agv.drive(STEPS_TURN, -STEPS_TURN)
        agv.wait_for_stop(timeout=10)
        print("Fertig.")

    elif cmd == 'h':
        steps180 = STEPS_TURN * 2
        print(f"180° links drehen ({steps180} Steps) ...")
        agv.drive(-steps180, steps180)
        agv.wait_for_stop(timeout=15)
        print("Fertig. AGV sollte exakt umgekehrt zeigen.")

    elif cmd == '+':
        STEPS_TURN += 5
        spr = STEPS_TURN / (math.pi / 2)
        print(f"STEPS_TURN = {STEPS_TURN}  → STEPS_PER_RAD ≈ {spr:.1f}")

    elif cmd == '-':
        STEPS_TURN = max(10, STEPS_TURN - 5)
        spr = STEPS_TURN / (math.pi / 2)
        print(f"STEPS_TURN = {STEPS_TURN}  → STEPS_PER_RAD ≈ {spr:.1f}")

    elif cmd == ']':
        STEPS_TEST += 10
        print(f"STEPS_TEST = {STEPS_TEST}")

    elif cmd == '[':
        STEPS_TEST = max(10, STEPS_TEST - 10)
        print(f"STEPS_TEST = {STEPS_TEST}")

    elif cmd == 'v':
        VEL = min(100, VEL + 5)
        agv.set_max_velocity(VEL)
        print(f"VEL = {VEL}%")

    elif cmd == 's':
        VEL = max(5, VEL - 5)
        agv.set_max_velocity(VEL)
        print(f"VEL = {VEL}%")

    else:
        print("f b l r h  + - ] [  v s  q")

agv.disable()

spr = STEPS_TURN / (math.pi / 2)
print(f"\n=== Kalibrierungsergebnis ===")
print(f"  STEPS_PER_CELL = {STEPS_TEST}   → in control/kinematics.py eintragen")
print(f"  STEPS_PER_RAD  = {spr:.0f}     → in control/kinematics.py eintragen")
print(f"  (STEPS_TURN={STEPS_TURN} / (π/2) = {spr:.1f})")
