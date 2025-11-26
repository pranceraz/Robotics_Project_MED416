'''
    © 2025 Arnav Yadavilli. All rights reserved.
'''
import machine
import time
from machine import Pin

# Pin initialization
leftrear_a = Pin(12, Pin.OUT)
leftrear_b = Pin(13, Pin.OUT)
leftfront_a = Pin(8, Pin.OUT)
leftfront_b = Pin(7, Pin.OUT)

rightrear_a = Pin(4, Pin.OUT)
rightrear_b = Pin(5, Pin.OUT)
rightfront_a = Pin(10, Pin.OUT)     
rightfront_b = Pin(11, Pin.OUT)

# User-defined movement functions
def forward(sleep_time=None):
    try:
        leftrear_b.value(1)
        rightrear_a.value(1)
        #leftfront_a.value(1)
        #rightfront_a.value(1)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in forward: {e}")

def backward(sleep_time=None):
    try:
        leftrear_a.value(1)
        rightrear_b.value(1)
        #leftfront_b.value(1)
        #rightfront_b.value(1)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in backward: {e}")

def right(sleep_time=None):
    try:
        leftrear_b.value(1)
        leftfront_a.value(1)
        #rightrear_b.value(1)
        #rightfront_b.value(1)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in right: {e}")

def left(sleep_time=None):
    try:
        leftrear_b.value(1)
        leftfront_b.value(1)
        rightrear_a.value(1)
        rightfront_b.value(1)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in left: {e}")

def stop():
    # Stops all motors
    leftrear_a.value(0)
    leftrear_b.value(0)
    leftfront_a.value(0)
    leftfront_b.value(0)
    rightrear_a.value(0)
    rightrear_b.value(0)
    rightfront_a.value(0)
    rightfront_b.value(0)

def main():
    print("Hey, I'm starting")
    forward(2)
    backward(2)
    left(1.5)
    right(1.5)
    
    #right(2)
    stop()
    print("Sequence done")

if __name__ == "__main__":
    main()