import os
import json
import logging
from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
from apscheduler.schedulers.background import BackgroundScheduler
import requests

app = Flask(__name__)
CORS(app)

@app.context_processor
def context_processor():
    """ Store the globals in a Flask way """
    return dict()

globvars        = context_processor()
globvars['status'] = {}       

config_file_path = './config.json'

def load_config():
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as file:
            return json.load(file)
    return {}

def save_config(config):
    with open(config_file_path, 'w') as file:
        json.dump(config, file)

def set_param(param):
    output = request.form.get(param)
    if not output:
        raise ValueError(f"Parameter {param} must not be empty")
    return output

@app.route('/set', methods=['POST'])
@cross_origin()
def set():
    try:
        project = set_param('project')
        host = set_param('host')
        port = set_param('port')
        desc = set_param('description')
        config = load_config()
        config[project] = {'host': host, 'port': port, 'description': desc}
        save_config(config)
        return make_response({'message': 'Project code set successfully'}, 200)
    except Exception as e:
        logging.error("Error setting project code: %s", e)
        return make_response({'error': str(e)}, 400)
 
@app.route('/get', methods=['GET'])
@cross_origin()
def get():
    try:
        project = request.args.get('project')
        if not project:
            raise ValueError("Project parameter is required")
        config = load_config()
        configs = config.get(project)
        if not configs:
            raise ValueError("Project not found")
        return make_response(
            {'project': project,
             'host': configs['host'],
             'port': configs['port'],
             'description': configs['description']
             }, 200)
    except Exception as e:
        logging.error("Error retrieving project code: %s", e)
        return make_response({'error': str(e)}, 400)

@app.route('/get_all', methods=['GET'])
@cross_origin()
def get_all():
    try:
        config = load_config()
        return make_response(config, 200)
    except Exception as e:
        logging.error("Error retrieving project code: %s", e)
        return make_response({'error': str(e)}, 400)

@app.route('/get_status', methods=['GET'])
@cross_origin()
def get_status():
    try:
        config = load_config()
        status = globvars['status']
        return make_response([config,status], 200)
    except Exception as e:
        logging.error("Error retrieving status: %s", e)
        return make_response({'error': str(e)}, 400)

def check_services():
    config = load_config()
    status = []
    for project, details in config.items():
        try:
            response = requests.get(f"http://{details['host']}:{details['port']}/ping")
            if response.status_code == 200:
                globvars['status'][project] = {
                    'status' : 'up',
                    'timestamp': response.json()['answer']
                }
                logging.info(f"Service {project} is up")
            else:
                globvars['status'][project] = {
                    'status' : 'down',
                    'timestamp': 'unknown'
                }
                logging.warning(f"Service {project} is down")
        except requests.ConnectionError:
            globvars['status'][project] = {
                'status' : 'down',
                'timestamp': 'unknown'
            }
            logging.error(f"Service {project} is down")
    print(globvars)
    
scheduler = BackgroundScheduler()
scheduler.add_job(check_services, 'interval', seconds=20)
scheduler.start()

if __name__ == '__main__':
    try:
        app.run(port=8000, debug=False, host="0.0.0.0")
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
