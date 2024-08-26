import yaml

CONFIG_FILE_PATH = './config/config.yml'
def load_config(file_path = CONFIG_FILE_PATH):
    """Load YAML configuration from a file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Load the configuration
config = load_config(CONFIG_FILE_PATH)

# Example usage
MODEL_NAME = config.get('MODEL_NAME')
DATA_PATH = config.get('DATA_PATH')
VECTOR_STORE_PATH_LLAMA_INDEX = config.get('VECTOR_STORE_PATH_LLAMA_INDEX')
VECTOR_STORE_PATH_LLANG_CHAIN = config.get('VECTOR_STORE_PATH_LLANG_CHAIN')
TITLE = config.get('TITLE')
ASSISTANT_MESSAGE = config.get('ASSISTANT_MESSAGE')


if __name__ == "__main__":
    print("Configuration Loaded:")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"VECTOR_STORE_PATH_LLAMA_INDEX: {VECTOR_STORE_PATH_LLAMA_INDEX}")
    print(f"VECTOR_STORE_PATH_LLANG_CHAIN: {VECTOR_STORE_PATH_LLANG_CHAIN}")
    print(f"TITLE: {TITLE}")
    print(f"ASSISTANT_MESSAGE: {ASSISTANT_MESSAGE}")
