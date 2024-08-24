import argparse
import os
import json
import subprocess
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, RobertaTokenizer, RobertaForSequenceClassification
from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for the models and data
MODEL_PATH_GPT_NEO = './models/gpt-neo'
MODEL_PATH_CODEBERT = './models/codebert'
DATA_PATH = './data'

# Define version directories
VERSION_DIRS = {
    'version_1': './version_1',
    'version_2': './version_2',
    'version_3': './version_3'
}

# Ensure model directories exist
os.makedirs(MODEL_PATH_GPT_NEO, exist_ok=True)
os.makedirs(MODEL_PATH_CODEBERT, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
for version_dir in VERSION_DIRS.values():
    os.makedirs(version_dir, exist_ok=True)

def download_model(model_name, model_path):
    try:
        if not os.path.exists(model_path):
            logging.info(f"Downloading model {model_name}...")
            model_bin = hf_hub_download(model_name, "pytorch_model.bin", cache_dir=model_path)
            tokenizer_bin = hf_hub_download(model_name, "tokenizer.json", cache_dir=model_path)
            logging.info(f"Model {model_name} downloaded to {model_path}")
    except Exception as e:
        logging.error(f"Error downloading model {model_name}: {e}")

download_model("EleutherAI/gpt-neo-1.3B", MODEL_PATH_GPT_NEO)
download_model("microsoft/codebert-base", MODEL_PATH_CODEBERT)

# Initialize GPT-Neo
tokenizer_gpt_neo = GPT2Tokenizer.from_pretrained(MODEL_PATH_GPT_NEO)
model_gpt_neo = GPT2LMHeadModel.from_pretrained(MODEL_PATH_GPT_NEO)

# Initialize CodeBERT
tokenizer_codebert = RobertaTokenizer.from_pretrained(MODEL_PATH_CODEBERT)
model_codebert = RobertaForSequenceClassification.from_pretrained(MODEL_PATH_CODEBERT)

def initialize_documents():
    docs = ['analysis_doc.json', 'code_improvement_doc.json', 'troubleshooting_doc.json']
    for doc in docs:
        path = os.path.join(DATA_PATH, doc)
        if not os.path.isfile(path):
            with open(path, 'w') as f:
                json.dump({}, f)
                logging.info(f"Created document {doc}")

initialize_documents()

def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}")
        return None

def analyze_code(version_3_path):
    try:
        with open(version_3_path, 'r') as file:
            code = file.read()

        inputs = tokenizer_codebert(code, return_tensors="pt")
        with torch.no_grad():
            outputs = model_codebert(**inputs)
        analysis = outputs.logits.argmax(dim=1).tolist()

        analysis_doc_path = os.path.join(DATA_PATH, 'analysis_doc.json')
        with open(analysis_doc_path, 'r+') as f:
            doc = json.load(f)
            doc[version_3_path] = analysis
            f.seek(0)
            json.dump(doc, f, indent=4)
        logging.info(f"Code analyzed and saved to {analysis_doc_path}")
    except Exception as e:
        logging.error(f"Error analyzing code: {e}")

def improve_code(analysis_doc_path, version_3_path):
    try:
        with open(analysis_doc_path, 'r') as f:
            analysis = json.load(f).get(version_3_path, {})

        prompt = f"Improve this code based on the following analysis: {analysis}"
        inputs = tokenizer_gpt_neo(prompt, return_tensors="pt")
        outputs = model_gpt_neo.generate(**inputs)
        improved_code = tokenizer_gpt_neo.decode(outputs[0])

        improved_code_path = os.path.join(DATA_PATH, 'code_improvement_doc.json')
        with open(improved_code_path, 'r+') as f:
            doc = json.load(f)
            doc[version_3_path] = improved_code
            f.seek(0)
            json.dump(doc, f, indent=4)
        logging.info(f"Code improved and saved to {improved_code_path}")
    except Exception as e:
        logging.error(f"Error improving code: {e}")

def test_code(version_3_path, improvement_doc_path):
    try:
        with open(improvement_doc_path, 'r') as f:
            improved_code = json.load(f).get(version_3_path, '')

        improved_code_path = os.path.join(VERSION_DIRS['version_3'], 'improved_code.py')
        with open(improved_code_path, 'w') as f:
            f.write(improved_code)

        result = subprocess.run(['python', improved_code_path], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Code tested successfully: {result.stdout}")
        else:
            logging.error(f"Code testing failed: {result.stderr}")

        troubleshooting_doc_path = os.path.join(DATA_PATH, 'troubleshooting_doc.json')
        with open(troubleshooting_doc_path, 'r+') as f:
            doc = json.load(f)
            doc[version_3_path] = {
                "result": result.stdout,
                "errors": result.stderr
            }
            f.seek(0)
            json.dump(doc, f, indent=4)
    except Exception as e:
        logging.error(f"Error testing code: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CI/CD pipeline stages.")
    parser.add_argument('--
