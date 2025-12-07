#!/usr/bin/env python3
"""
GPIO Handler for Physical Button Control

This module handles physical GPIO button interactions on Raspberry Pi.
It monitors GPIO pin state and updates SharedState accordingly.
"""

import logging
import time
import threading
from .utils import SharedState

try:
    from gpiozero import Button
    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False
    logging.warning("gpiozero not installed. GPIO button will not be available.")


class GPIOHandler:
    """
    Manages physical GPIO button for Raspberry Pi.
    Monitors button press/release and updates SharedState.
    """
    
    def __init__(self, button_pin: int, shared_state: SharedState, pull_up: bool = True, bounce_time: float = 0.02):
        """
        Initialize GPIO handler.
        
        Args:
            button_pin: GPIO pin number (BCM numbering)
            shared_state: SharedState instance to update button state
            pull_up: Whether to use pull-up resistor (True for pull-up, False for pull-down)
            bounce_time: Debounce time in seconds
        """
        self.button_pin = button_pin
        self.shared_state = shared_state
        self.pull_up = pull_up
        self.bounce_time = bounce_time
        self.button = None
        self.is_running = False
        self.monitor_thread = None
        
        if not GPIOZERO_AVAILABLE:
            logging.error("gpiozero is not installed. Cannot initialize GPIO handler.")
            return
        
        try:
            # Initialize button with gpiozero
            self.button = Button(
                pin=button_pin,
                pull_up=pull_up,
                bounce_time=bounce_time,
                hold_time=0.1
            )
            
            # Attach event handlers
            self.button.when_pressed = self._on_button_pressed
            self.button.when_released = self._on_button_released
            
            logging.info(f"GPIO Handler initialized on pin {button_pin}")
            
        except Exception as e:
            logging.error(f"Failed to initialize GPIO button on pin {button_pin}: {e}")
            self.button = None
    
    def _on_button_pressed(self):
        """Callback when button is pressed."""
        logging.debug(f"Button pressed (pin {self.button_pin})")
        self.shared_state.set_button_state(True)
    
    def _on_button_released(self):
        """Callback when button is released."""
        logging.debug(f"Button released (pin {self.button_pin})")
        self.shared_state.set_button_state(False)
    
    def start(self):
        """Start monitoring GPIO button (event-driven, no thread needed with gpiozero)."""
        if self.button is None:
            logging.error("Button not initialized. Cannot start GPIO handler.")
            return False
        
        self.is_running = True
        logging.info(f"GPIO Handler started on pin {self.button_pin}")
        return True
    
    def stop(self):
        """Stop monitoring GPIO button and cleanup."""
        if self.button is None:
            return
        
        self.is_running = False
        try:
            self.button.close()
            logging.info(f"GPIO Handler stopped and cleaned up (pin {self.button_pin})")
        except Exception as e:
            logging.error(f"Error during GPIO cleanup: {e}")
    
    def get_button_state(self) -> bool:
        """Get current button state from GPIO."""
        if self.button is None:
            return False
        
        try:
            # gpiozero button is_pressed is True when button is physically pressed
            return self.button.is_pressed
        except Exception as e:
            logging.error(f"Error reading button state: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Alternative implementation using RPi.GPIO (if gpiozero is not available)
class GPIOHandlerRPiGPIO:
    """
    Fallback GPIO handler using RPi.GPIO library.
    Use this if gpiozero is not available.
    """
    
    def __init__(self, button_pin: int, shared_state: SharedState, pull_up: bool = True):
        """
        Initialize GPIO handler using RPi.GPIO.
        
        Args:
            button_pin: GPIO pin number (BCM numbering)
            shared_state: SharedState instance to update button state
            pull_up: Whether to use pull-up resistor
        """
        self.button_pin = button_pin
        self.shared_state = shared_state
        self.pull_up = pull_up
        self.is_running = False
        self.monitor_thread = None
        
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            
            # Setup GPIO
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setup(button_pin, self.GPIO.IN, pull_up_down=self.GPIO.PUD_UP if pull_up else self.GPIO.PUD_DOWN)
            
            logging.info(f"GPIO Handler (RPi.GPIO) initialized on pin {button_pin}")
            
        except ImportError:
            logging.error("RPi.GPIO is not installed. Cannot initialize GPIO handler.")
            self.GPIO = None
        except Exception as e:
            logging.error(f"Failed to initialize GPIO button on pin {button_pin}: {e}")
            self.GPIO = None
    
    def _monitor_button(self):
        """Monitor button in a separate thread."""
        previous_state = False
        
        while self.is_running:
            try:
                # Read button state
                current_state = self.GPIO.input(self.button_pin) == 0  # 0 means pressed with pull-up
                
                # Only update if state changed
                if current_state != previous_state:
                    self.shared_state.set_button_state(current_state)
                    
                    if current_state:
                        logging.debug(f"Button pressed (pin {self.button_pin})")
                    else:
                        logging.debug(f"Button released (pin {self.button_pin})")
                    
                    previous_state = current_state
                
                time.sleep(0.01)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logging.error(f"Error in button monitor thread: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start monitoring GPIO button."""
        if self.GPIO is None:
            logging.error("GPIO not initialized. Cannot start handler.")
            return False
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_button, daemon=True)
        self.monitor_thread.start()
        logging.info(f"GPIO Handler started on pin {self.button_pin}")
        return True
    
    def stop(self):
        """Stop monitoring GPIO button and cleanup."""
        if self.GPIO is None:
            return
        
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        try:
            self.GPIO.cleanup(self.button_pin)
            logging.info(f"GPIO Handler stopped and cleaned up (pin {self.button_pin})")
        except Exception as e:
            logging.error(f"Error during GPIO cleanup: {e}")
    
    def get_button_state(self) -> bool:
        """Get current button state from GPIO."""
        if self.GPIO is None:
            return False
        
        try:
            return self.GPIO.input(self.button_pin) == 0
        except Exception as e:
            logging.error(f"Error reading button state: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_gpio_handler(button_pin: int, shared_state: SharedState, use_gpiozero: bool = True) -> object:
    """
    Factory function to create appropriate GPIO handler.
    
    Args:
        button_pin: GPIO pin number (BCM numbering)
        shared_state: SharedState instance
        use_gpiozero: Prefer gpiozero if available
    
    Returns:
        GPIO handler instance (GPIOHandler or GPIOHandlerRPiGPIO)
    """
    if use_gpiozero and GPIOZERO_AVAILABLE:
        return GPIOHandler(button_pin, shared_state)
    else:
        return GPIOHandlerRPiGPIO(button_pin, shared_state)
