import os
import json
import logging
from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

config_file_path = '/home/leen/projects/rag/config.json'

def load_config():
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as file:
            return json.load(file)
    return {}

def save_config(config):
    with open(config_file_path, 'w') as file:
        json.dump(config, file)

@app.route('/set', methods=['POST'])
@cross_origin()
def set():
    try:
        project = request.form['project']
        if not project:
            raise ValueError("Project code must not be empty")
        api = request.form['api']
        if not api:
            raise ValueError("API host must not be empty")
        config = load_config()
        config[project] = api
        save_config(config)
        return make_response({'message': 'Project code set successfully'}, 200)
    except Exception as e:
        logging.error("Error setting project code: %s", e)
        return make_response({'error': str(e)}, 400)

@app.route('/get', methods=['GET'])
@cross_origin()
def get():
    try:
        project = request.args['project']
        config = load_config()
        api = config[project]
        return make_response(
            {'project': project,
             'api': api}, 200)
    except Exception as e:
        logging.error("Error retrieving project code: %s", e)
        return make_response({'error': str(e)}, 400)

if __name__ == '__main__':
    app.run(port=5001, debug=False, host="0.0.0.0")
