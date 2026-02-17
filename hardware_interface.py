
import serial
import time
import sys

class ArduinoController:
    """
    Handles communication with Arduino for page flipping hardware.
    Protocol:
        - Python sends: "flip" (triggers hardware action)
        - Arduino sends: "capture" (confirms action complete, ready for next capture)
    """
    
    def __init__(self, port: str = "COM4", baud_rate: int = 9600, timeout: int = 30):
        """
        Initialize Serial Connection
        
        Args:
            port: COM port (e.g., COM4)
            baud_rate: Baud rate (default 9600)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.connection = None
        
        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            # Wait for Arduino to reset/initialize
            time.sleep(2)
            print(f"[Hardware] Connected to Arduino on {self.port}")
            
        except serial.SerialException as e:
            print(f"[Hardware] ERROR: Could not connect to {self.port}")
            print(f"[Hardware] Details: {e}")
            raise e

    def send_flip_signal(self):
        """Send 'flip' command to Arduino"""
        if not self.connection:
            return
            
        try:
            command = "flip\n"
            self.connection.write(command.encode('utf-8'))
            print("[Hardware] Sent 'flip' signal")
        except Exception as e:
            print(f"[Hardware] ERROR sending flip signal: {e}")

    def wait_for_capture_signal(self) -> bool:
        """
        Block and wait for 'capture' command from Arduino.
        Returns True if signal received, False if timeout or error.
        """
        if not self.connection:
            return False
            
        print("[Hardware] Waiting for 'capture' confirmation...")
        start_time = time.time()
        
        try:
            while True:
                if self.connection.in_waiting > 0:
                    line = self.connection.readline().decode('utf-8').strip()
                    if line == "capture":
                        print("[Hardware] Received 'capture' signal!")
                        return True
                    else:
                        # Ignore other debug messages from Arduino
                        # print(f"[Hardware Debug]: {line}")
                        pass
                
                # Check for overall timeout
                if (time.time() - start_time) > self.timeout:
                    print("[Hardware] ERROR: Timeout waiting for capture signal")
                    return False
                    
                time.sleep(0.1)
                
        except Exception as e:
            print(f"[Hardware] ERROR reading from serial: {e}")
            return False

    def close(self):
        """Close serial connection"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("[Hardware] Connection closed")

if __name__ == "__main__":
    # Simple test
    try:
        controller = ArduinoController("COM4")
        controller.send_flip_signal()
        if controller.wait_for_capture_signal():
            print("Test Success!")
        else:
            print("Test Failed/Timeout")
        controller.close()
    except Exception as e:
        print("Test Aborted")
