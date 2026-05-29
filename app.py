import os
import base64

from flask import Flask, request, jsonify, render_template

from inference import evaluate_image

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json(silent=True) or {}
    image_field = data.get('image')
    if not image_field:
        return jsonify({"error": "No image provided."}), 400

    # Strip a possible "data:image/png;base64," prefix.
    if ',' in image_field:
        image_field = image_field.split(',', 1)[1]

    try:
        image_bytes = base64.b64decode(image_field)
    except Exception:
        return jsonify({"error": "Invalid image encoding."}), 400

    try:
        result = evaluate_image(image_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Failed to evaluate the expression."}), 400

    return jsonify(result), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    app.run(host='0.0.0.0', port=port)
