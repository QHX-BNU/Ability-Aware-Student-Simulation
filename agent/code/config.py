import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path: stores files required for simulation and LLM configuration
DATA_PATH = os.path.join(BASE_DIR, '../data')

RESULT_PATH_ROUTER = os.path.join(BASE_DIR, 'router')


BASE_URL_GPT_5_MINI = 'https://api.openai.com/v1/chat/completions'
OPENAI_API_KEY_GPT_5_MINI = 'sk-xxxxxx'
MODEL_GPT_5_MINI = 'gpt-5-mini'
# Simulation results output directory
RESULTS_DIR_GPT_5_MINI = os.path.join(BASE_DIR, 'gpt5mini')


BASE_URL_LLAMA_3b = 'local_path_to_llama_3b_model'
OPENAI_API_KEY_LLAMA_3b = 'local'
MODEL_LLAMA_3b = 'Llama-3.2-3B-Instruct'
RESULTS_DIR_LLAMA_3b = os.path.join(BASE_DIR, 'llama_3b')

# Simulation parameters
SIM_PARAMS = {
    'memory_source':    'real',
    'learning_effect':  'yes',
    'forgetting_effect':'yes',
    'reflection_choice':'yes',
    'sim_strategy':     'performance',
    # 'gpt_type':         0,  # 0: GPT-3.5, 1: GPT-4
    'short_term_size':  3,
    'long_term_thresh': 5,
    'forget_lambda':    0.99
}
