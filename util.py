import json
import os
from pathlib import Path
from typing import Any, Dict

from model import load_embedding_from_config, load_llm_from_config
from setting import Settings
import logging
from pythonjsonlogger import jsonlogger



def verify_openai_token(token: str) -> str:
    import openai

    openai.api_key = token
    try:
        openai.Completion.create(
            model="text-ada-001",
            prompt="Hello",
            temperature=0,
            max_tokens=10,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0,
        )
        return "OK"
    except Exception as e:
        return str(e)

def get_logging(logger_name,content=''):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logHandlerJson = logging.FileHandler('./memory_data/'+logger_name+'.json')
        formatter = jsonlogger.JsonFormatter()
        logHandlerJson.setFormatter(formatter)

        # handler = logging.FileHandler('./memory_data/'+logger_name+'.txt')
        # handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(logHandlerJson)
    logger.info(content)


def verify_model_initialization(settings: Settings) -> str:
    try:
        load_llm_from_config(settings.model.llm)
    except Exception as e:
        return f"LLM initialization check failed: {e}"

    try:
        load_embedding_from_config(settings.model.embedding)
    except Exception as e:
        return f"Embedding initialization check failed: {e}"

    return "OK"


def verify_pinecone_token(token: str) -> str:
    return "OK"


def verify_discord_token(token: str) -> str:
    return "OK"


def load_json_value(filepath: Path, key: str, default_value: Any) -> Any:
    if not Path(filepath).exists():
        return default_value
    json_obj = load_json(filepath)
    if key not in json_obj:
        return default_value
    return json_obj[key]


def set_json_value(filepath: Path, key: str, value: Any) -> None:
    # key needs to follow python naming convention, such as trial_id
    json_obj = load_json(filepath)
    json_obj[key] = value
    with open(filepath, "w+") as json_file:
        json.dump(json_obj, json_file, sort_keys=True)
        json_file.flush()


def load_json(filepath: Path) -> Dict:
    if not Path(filepath).exists():
        return {}
    with open(filepath, "r") as file:
        try:
            json_obj = json.load(file)
            return json_obj
        except json.JSONDecodeError as e:
            if os.stat(filepath).st_size == 0:
                # Empty file
                return {}
            else:
                raise e

def load_log(file_name, key_name):
    content_list = []
    key_list = []
    with open('./memory_data/'+file_name) as f:
        contents = f.readlines()
        for i in contents:
            print(i)
            contents =  json.loads(i)
            content_list.append(list(contents.values())[1][key_name])
            key_list.append(list(contents.keys())[1])
    return content_list, key_list

def load_log_full(file_name, key_name):
    content_list = []
    key_list = []
    with open(file_name) as f:
        contents = f.readlines()
        for i in contents:
            #print(i)
            contents = json.loads(i)
            if key_name is None:
                content_list.append(list(contents.values())[1])
            else:
                content_list.append(list(contents.values())[1][key_name])
            key_list.append(list(contents.keys())[1])
    return content_list, key_list

def get_checkpoint_dir(agent_file: str) -> str:
    return "./{}.cpt".format(os.path.basename(agent_file))
