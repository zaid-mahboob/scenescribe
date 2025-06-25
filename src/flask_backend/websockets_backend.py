from flask import Flask, request, jsonify
import threading
from ..scenescribe.scenescribe import SceneScribe
from ..lib.utils import SharedState

sharedState = SharedState()
scenescribe = SceneScribe(shared_state = sharedState, language='urdu')

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

if __name__ == '__main__':
    # Run on SoftAP IP (192.168.4.1) port 5000
    threading.Thread(target=scenescribe.run).start()
    app.run(host='192.168.4.1', port=5000)