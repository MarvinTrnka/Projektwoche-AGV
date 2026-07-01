#!/bin/bash
# Prüft ob AGV und Kamera erreichbar sind

AGV_IP="172.17.1.88"
CAM_URL="http://10.250.150.224:8081/"

echo "Prüfe Verbindungen..."

if curl -s --connect-timeout 3 "http://$AGV_IP/api/agv/stepper/isMoving" | grep -q "200"; then
    echo "✓ AGV erreichbar auf $AGV_IP"
else
    echo "✗ AGV nicht gefunden auf $AGV_IP"
fi

if curl -s --connect-timeout 3 -I "$CAM_URL" 2>&1 | grep -q "200"; then
    echo "✓ Kamera erreichbar"
else
    echo "✗ Kamera nicht erreichbar"
fi

echo ""
echo "Starte: cd Makerweek && python3 Main.py"
