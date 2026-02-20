import serial
import time
s = serial.Serial("COM3", 115200)
s.write(b"\r\n\r\n")
time.sleep(2)
gcode = """G21
G17
G90
G00 X0.0000 Y297.0000
G01 X0.0000 Y197.0000
G01 X100.0000 Y197.0000
G01 X100.0000 Y297.0000
G01 X0.0000 Y297.0000
M2
"""
s.write(gcode.encode())
print("Sent move command")
s.close()