import os
import json
import logging
import subprocess
from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
from apscheduler.schedulers.background import BackgroundScheduler
import requests

logging.basicConfig(level='ERROR')

app = Flask(__name__)
CORS(app)

@app.context_processor
def context_processor():
    """ Store the globals in a Flask way """
    return dict()

globvars = context_processor()
globvars['processes'] = {}       

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
    output = request.json.get(param)
    if not output:
        raise ValueError(f"Parameter {param} must not be empty")
    return output

@app.route('/start', methods=['GET'])
@cross_origin()
def start():
    project = request.args.get('project')
    globvars['processes'][project] = subprocess.Popen(['python','ragservice.py',project])
    return make_response({'message':'Project started'}, 200)    

@app.route('/stop', methods=['GET'])
@cross_origin()
def stop():
    project = request.args.get('project')
    p = globvars['processes'][project]
    return make_response({'message':p.kill()},200)

@app.route('/set', methods=['POST'])
@cross_origin()
def set():
    my_input = request.json
    my_keys  = ['project','port','description']
    project  = my_input['project']
    if 'originalProject' in my_input:
        original_project = my_input['originalProject']
    print(my_input)
    try:
        config = load_config()
        update = {}
        for key in my_keys:
            update.update({key: my_input[key]})

        if project not in config:
            config[project] = {}
        config[project] = update
        if 'originalProject' in my_input:
            if original_project != project:
                config.pop(original_project)
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

@app.route('/delete', methods=['DELETE'])
@cross_origin()
def delete():
    try:
        project = request.args.get('project')
        if not project:
            raise ValueError("Project parameter is required")
        config = load_config()
        if project in config:
            del config[project]
            save_config(config)
            return make_response({'message': 'Project deleted successfully'}, 200)
        else:
            raise ValueError("Project not found")
    except Exception as e:
        logging.error("Error deleting project: %s", e)
        return make_response({'error': str(e)}, 400)

def check_services():
    config = load_config()
    for project, details in config.items():
        try:
            response = requests.get(f"http://localhost:{details['port']}/ping")
            if response.status_code == 200:
                resp = response.json()['answer']
                config[project].update({
                    'status' : 'up',
                    'timestamp': resp['timestamp'],
                    'pid': resp['pid']
                })
                #logging.info(f"Service {project} is up")
            else:
                config[project].update({
                    'status' : 'down',
                    'timestamp': 'unknown'
                })
                #logging.warning(f"Service {project} is down")
        except requests.ConnectionError:
            config[project].update({
                'status' : 'down',
                'timestamp': 'unknown'
            })
            #logging.error(f"Service {project} is down")
    save_config(config)
    
scheduler = BackgroundScheduler()
scheduler.add_job(check_services, 'interval', seconds=10)
scheduler.start()

if __name__ == '__main__':
    try:
        app.run(port=8000, debug=False, host="0.0.0.0")
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
