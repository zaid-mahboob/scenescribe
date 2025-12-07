#!/usr/bin/env python3
"""
GPIO Button Configuration and Usage Guide

This file demonstrates how to configure and use the GPIO button handler
for your Raspberry Pi SceneScribe setup.
"""

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# In src/flask_backend/websockets_backend.py:
# - GPIO_BUTTON_PIN: Set this to your GPIO pin number (BCM numbering)
# - USE_GPIO_BUTTON: Set to True to enable, False to disable GPIO button

GPIO_BUTTON_PIN = 17  # Change to your pin (e.g., 17, 27, 22, etc.)
USE_GPIO_BUTTON = True


# ==============================================================================
# GPIO PIN REFERENCE (Raspberry Pi)
# ==============================================================================

"""
Common GPIO pins (BCM numbering):
- GPIO 17: Pin 11 (default in this implementation)
- GPIO 27: Pin 13
- GPIO 22: Pin 15
- GPIO 23: Pin 16
- GPIO 24: Pin 18
- GPIO 25: Pin 22

Use BCM numbering (not physical pin numbers)
"""


# ==============================================================================
# HARDWARE SETUP
# ==============================================================================

"""
Wiring with Pull-Up Resistor (Recommended):

    3.3V
     |
     R (10K ohm resistor)
     |
    GPIO Pin ----+---- GND (when button pressed)
     |           |
     +---[Button]--+

OR with Raspberry Pi's Internal Pull-Up:
    3.3V
     |
    [Button]
     |
    GPIO Pin (internal pull-up enabled)
"""


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

# Example 1: Using GPIO handler in websockets_backend.py (default setup)
"""
The GPIO handler is automatically initialized when you run:
    python -m src.flask_backend.websockets_backend

It will:
- Monitor the GPIO pin for button presses
- Update SharedState when button state changes
- Work seamlessly with existing recording methods
"""

# Example 2: Manual initialization
"""
from src.lib.gpio_handler import create_gpio_handler
from src.lib.utils import SharedState

shared_state = SharedState()
gpio_handler = create_gpio_handler(
    button_pin=17,
    shared_state=shared_state,
    use_gpiozero=True  # Set to False to use RPi.GPIO fallback
)

gpio_handler.start()

# Your code here...

gpio_handler.stop()  # Clean up on exit
"""

# Example 3: Using context manager
"""
from src.lib.gpio_handler import create_gpio_handler
from src.lib.utils import SharedState

shared_state = SharedState()

with create_gpio_handler(17, shared_state) as gpio_handler:
    # GPIO button is active in this block
    while True:
        if shared_state.get_button_state():
            print("Button is pressed!")
        time.sleep(0.1)
"""


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

"""
1. Check GPIO Status:
   GET http://192.168.4.1:5001/api/gpio/status
   
   Response:
   {
       "enabled": true,
       "pin": 17,
       "current_state": false,
       "handler_type": "gpiozero"
   }

2. Manually Set Button State (SoftAP replacement):
   POST http://192.168.4.1:5001/api/state
   Body: {"state": true}  // or false

3. Test Endpoint:
   POST http://192.168.4.1:5001/test
   Body: {"data": "test"}
"""


# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

"""
1. "gpiozero not installed" warning:
   - Install: pip install gpiozero
   - Falls back to RPi.GPIO automatically
   
2. "Permission denied" error:
   - Run with sudo: sudo python -m src.flask_backend.websockets_backend
   - Or configure GPIO permissions in /etc/sudoers
   
3. Button not responding:
   - Check GPIO pin number (use `gpio readall` to verify)
   - Test with: python -c "from gpiozero import Button; b = Button(17); print(b.is_pressed)"
   - Check wiring and resistor connections
   
4. GPIO already in use:
   - Another process is using the GPIO
   - Use `gpio unexport 17` to release (replace 17 with your pin)
"""


# ==============================================================================
# SWITCHING BETWEEN GPIO AND SOFTAP
# ==============================================================================

"""
GPIO Button (Hardware):
- Set USE_GPIO_BUTTON = True in websockets_backend.py
- Uses physical GPIO button

SoftAP Button (Network):
- Set USE_GPIO_BUTTON = False in websockets_backend.py
- Use POST /api/state endpoint to control

Both can run simultaneously - whichever updates SharedState wins
"""


# ==============================================================================
# INTEGRATION WITH RECORDING
# ==============================================================================

"""
The GPIO button automatically works with:

1. record_with_manual_button() in src/lib/utils.py
   - Listens to SharedState.button_on
   - Records audio while button is pressed
   
2. SceneScribe.run() main loop
   - Can trigger commands on button press
   - Uses shared_state.get_button_state()

No changes needed to existing code - GPIO updates SharedState!
"""
