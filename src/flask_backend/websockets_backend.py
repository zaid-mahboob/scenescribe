from flask import Flask, request, jsonify
import threading
import logging
from ..scenescribe.scenescribe import SceneScribe
from ..lib.utils import SharedState
from ..lib.gpio_handler import create_gpio_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPIO Configuration
GPIO_BUTTON_PIN = 17  # Change this to your actual GPIO pin (BCM numbering)
USE_GPIO_BUTTON = True  # Set to False to disable GPIO button

sharedState = SharedState()
scenescribe = SceneScribe(shared_state = sharedState, language='urdu')
gpio_handler = None

# Initialize GPIO handler if enabled
if USE_GPIO_BUTTON:
    try:
        gpio_handler = create_gpio_handler(GPIO_BUTTON_PIN, sharedState, use_gpiozero=True)
        gpio_handler.start()
        logger.info(f"GPIO button handler started on pin {GPIO_BUTTON_PIN}")
    except Exception as e:
        logger.warning(f"Failed to initialize GPIO button: {e}. Continuing without GPIO support.")
        gpio_handler = None

app = Flask(__name__)
@app.route('/test', methods=['POST', 'GET'])
def test_endpoint():
    data = request.get_json()
    print(f"Received data: {data}")
    return jsonify({"status": "success", "data": data})

@app.route('/api/state', methods=['POST'])
def handle_state():
    data = request.get_json()
    state = data.get('state')
    print(f"Received state: {state}")
    sharedState.set_button_state(state)
    return jsonify({"status": "success", "state": state})

@app.route('/api/gpio/status', methods=['GET'])
def get_gpio_status():
    """Get GPIO button status and configuration."""
    gpio_status = {
        "enabled": USE_GPIO_BUTTON,
        "pin": GPIO_BUTTON_PIN if USE_GPIO_BUTTON else None,
        "current_state": gpio_handler.get_button_state() if gpio_handler else None,
        "handler_type": "gpiozero" if gpio_handler and hasattr(gpio_handler, 'button') else "rpi_gpio" if gpio_handler else None
    }
    return jsonify(gpio_status)

def cleanup_gpio():
    """Cleanup GPIO resources on shutdown."""
    global gpio_handler
    if gpio_handler:
        try:
            gpio_handler.stop()
            logger.info("GPIO handler cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during GPIO cleanup: {e}")

if __name__ == '__main__':
    # Run on SoftAP IP (192.168.4.1) port 5000
    threading.Thread(target=scenescribe.run).start()
    app.run(host='192.168.4.1', port=5001)