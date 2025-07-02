from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/submit", methods=["POST"])
def submit_credentials():
    ssid = request.form.get("ssid")
    password = request.form.get("password")

    if not ssid or not password:
        return jsonify(
            {"status": "error", "message": "Missing credentials"}), 400

    print(f"Received credentials: SSID={ssid}, Password={password}")

    # TODO: Save to wpa_supplicant.conf or pass to ESP32

    return jsonify(
        {"status": "success", "message": "Credentials received!"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
