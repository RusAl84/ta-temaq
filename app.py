from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin
from tmodel import tmodel


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def dafault_route():
    return 'API'


@app.route('/tmodel', methods=['POST'])
def rtmodel():
    data = request.json
    text = data['text']
    topic_num = data['topic_num']
    print(text)
    print(topic_num)
    tmodel(text, topic_num)
    return jsonify("ok"), 200, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5003")
# app.run(host="0.0.0.0")
