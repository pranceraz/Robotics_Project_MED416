'''
    © 2025 Arnav Yadavilli. All rights reserved.
'''
import RPi.GPIO as GPIO
import time


# BCM is recommended as it refers to the GPIO channel, not the physical pin number.
GPIO.setmode(GPIO.BCM) 

# --- PIN INITIALIZATION ---
# Assign BCM pin numbers (These are the GPIO numbers, NOT the physical pin numbers)
# NOTE: These pin numbers must match where you physically wired the Raspberry Pi.

# Left Side Motor Driver Input Pins
LEFT_REAR_A = 22
LEFT_REAR_B = 23
LEFT_FRONT_A = 17
LEFT_FRONT_B = 18

# Right Side Motor Driver Input Pins
RIGHT_REAR_A = 12
RIGHT_REAR_B = 16
RIGHT_FRONT_A = 24
RIGHT_FRONT_B = 25

# Group all pins together
PINS = [
    LEFT_REAR_A, LEFT_REAR_B, LEFT_FRONT_A, LEFT_FRONT_B,
    RIGHT_REAR_A, RIGHT_REAR_B, RIGHT_FRONT_A, RIGHT_FRONT_B
]

# Set all pins as OUTPUT
for pin in PINS:
    GPIO.setup(pin, GPIO.OUT)

# --- USER-DEFINED MOVEMENT FUNCTIONS ---

def forward(sleep_time=None):
    """Moves the robot straight forward."""
    try:
        # Left Side: A=LOW, B=HIGH (or vice versa, depending on wiring)
        # Right Side: A=HIGH, B=LOW
        
        # Original logic: leftrear_b=1, rightrear_a=1
        # To make a 4-wheel drive, let's assume all 'B' pins are LOW and all 'A' pins are HIGH for one direction (or vice versa).
        
        # Turn all motors ON in the forward direction
        GPIO.output(LEFT_REAR_A, GPIO.LOW)
        GPIO.output(LEFT_REAR_B, GPIO.HIGH)
        GPIO.output(LEFT_FRONT_A, GPIO.LOW)
        GPIO.output(LEFT_FRONT_B, GPIO.HIGH)
        
        GPIO.output(RIGHT_REAR_A, GPIO.HIGH) # Right side needs inverse logic
        GPIO.output(RIGHT_REAR_B, GPIO.LOW)
        GPIO.output(RIGHT_FRONT_A, GPIO.HIGH)
        GPIO.output(RIGHT_FRONT_B, GPIO.LOW)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in forward: {e}")

def backward(sleep_time=None):
    """Moves the robot straight backward."""
    try:
        # Turn all motors ON in the backward direction
        GPIO.output(LEFT_REAR_A, GPIO.HIGH)
        GPIO.output(LEFT_REAR_B, GPIO.LOW)
        GPIO.output(LEFT_FRONT_A, GPIO.HIGH)
        GPIO.output(LEFT_FRONT_B, GPIO.LOW)
        
        GPIO.output(RIGHT_REAR_A, GPIO.LOW) # Right side needs inverse logic
        GPIO.output(RIGHT_REAR_B, GPIO.HIGH)
        GPIO.output(RIGHT_FRONT_A, GPIO.LOW)
        GPIO.output(RIGHT_FRONT_B, GPIO.HIGH)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in backward: {e}")

def right(sleep_time=None):
    """Turns the robot right (by rotating left wheels forward and right wheels backward)."""
    try:
        # Left wheels forward
        GPIO.output(LEFT_REAR_A, GPIO.LOW)
        GPIO.output(LEFT_REAR_B, GPIO.HIGH)
        GPIO.output(LEFT_FRONT_A, GPIO.LOW)
        GPIO.output(LEFT_FRONT_B, GPIO.HIGH)
        
        # Right wheels backward
        GPIO.output(RIGHT_REAR_A, GPIO.LOW)
        GPIO.output(RIGHT_REAR_B, GPIO.HIGH)
        GPIO.output(RIGHT_FRONT_A, GPIO.LOW)
        GPIO.output(RIGHT_FRONT_B, GPIO.HIGH)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in right: {e}")

def left(sleep_time=None):
    """Turns the robot left (by rotating right wheels forward and left wheels backward)."""
    try:
        # Left wheels backward
        GPIO.output(LEFT_REAR_A, GPIO.HIGH)
        GPIO.output(LEFT_REAR_B, GPIO.LOW)
        GPIO.output(LEFT_FRONT_A, GPIO.HIGH)
        GPIO.output(LEFT_FRONT_B, GPIO.LOW)
        
        # Right wheels forward
        GPIO.output(RIGHT_REAR_A, GPIO.HIGH)
        GPIO.output(RIGHT_REAR_B, GPIO.LOW)
        GPIO.output(RIGHT_FRONT_A, GPIO.HIGH)
        GPIO.output(RIGHT_FRONT_B, GPIO.LOW)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in left: {e}")

def stop():
    """Stops all motors by setting all control pins to LOW."""
    for pin in PINS:
        GPIO.output(pin, GPIO.LOW)

def main():
    try:
        print("Hey, I'm starting the motor sequence.")
        forward(2)
        backward(2)
        left(1.5)
        right(1.5)
        
        stop()
        print("Sequence done.")
        
    except KeyboardInterrupt:
        print("\nSequence interrupted.")
        
    finally:
        # --- CRITICAL: Cleanup GPIO pins ---
        GPIO.cleanup()
        print("GPIO cleanup complete.")

if __name__ == "__main__":
    main()