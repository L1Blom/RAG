import configparser
import os
import json
import logging
import subprocess
import sys
from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from urllib.parse import urlparse
import threading

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

@app.context_processor
def context_processor():
    """ Store the globals in a Flask way """
    return dict()

# Read the constants from a config file

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

def set_env(project):
    rc = configparser.ConfigParser()
    constantsfile = "constants/constants_"+project+".ini"
    if os.path.exists(constantsfile):
        rc.read(constantsfile)
    else: 
        rc.read("constants/constants.ini")  

    config = load_config()
    if project in config:
        rcdef                   = rc['DEFAULT']
        rcdef['id']             = project
        rcdef['data_dir']       = 'data/'+project
        rcdef['html']           = rcdef['data_dir']
        os.makedirs(rcdef['data_dir'], exist_ok=True)  # Ensure data directory exists
        rcdef['persistence']    = rcdef['data_dir']+'/vectorstore'
        rcflask                 = rc['FLASK']
        rcflask['port']         = config[project]['port']
        rcllm                   = rc['LLMS']
        rcllm['use_llm']        = config[project]['provider']
        newconstants = "constants/constants_"+project+".ini"
        with open(newconstants,'w') as nc:
            rc.write(nc)
    else:
        raise Exception("This project is not found: "+project)

def read_process_output(process):
    def target():
        for line in iter(process.stdout.readline, b''):
            print(line.decode(), end='')
    thread = threading.Thread(target=target)
    thread.start()

def load_configurations():
    rc = configparser.ConfigParser()
    config = load_config()
    for project in config:
        constantsfile = "constants/constants_"+project+".ini"
        if os.path.exists(constantsfile):
            rc.read(constantsfile)
        else:
            logging.error("Error: constants file not found for project "+project)
            continue
        config[project]['status'] = 'undefined'
        llms = rc['LLMS']['use_llm']
        config[project]['llm'] = rc['LLMS.'+llms]['modeltext']
        save_config(config)
        globvars['processes'][project] = subprocess.Popen(
            ['python', 'ragservice.py', project], 
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0)
        read_process_output(globvars['processes'][project])
        logging.info(f"Project {project} started")  


@app.route('/start', methods=['GET'])
@cross_origin()
def start():
    project = request.args.get('project')
    globvars['processes'][project] = subprocess.Popen(
        ['python', 'ragservice.py', project], 
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0)
    read_process_output(globvars['processes'][project])
    logging.info(f"Project {project} started")
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
    my_input  = request.json
    print(my_input)
    my_keys   = ['project','port','description','provider','llm']  
    project   = my_input['project']
    if 'originalProject' in my_input:
        original_project = my_input['originalProject']
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
        set_env(project)
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
             'port': configs['port'],
             'provider': configs['provider'],
             'llm': configs['llm'],
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
    in_docker = os.environ.get('IN_DOCKER', False)
    hostname = os.environ.get('REACT_APP_RAG_SERVER') or 'localhost'
            
    for project, details in config.items():
        host = 'http://'+hostname+':'+details['port']
        try:
            response = requests.get(f"{host}/ping")
            if response.status_code == 200:
                resp = response.json()['answer']
                config[project].update({
                    'status' : 'up',
                    'timestamp': resp['timestamp'],
                    'llm': resp['llm']
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
        load_configurations()
        app.run(port=8000, debug=False, host="0.0.0.0")
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
