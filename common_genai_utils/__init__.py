from dotenv import load_dotenv

def load_env():
    loaded_env = load_dotenv(dotenv_path=".env", override=True)
    print("Loaded env variables from current directory -> ", loaded_env)
    if loaded_env == False:
        loaded_env = load_dotenv(dotenv_path="../.env", override=True)
        print("Loaded env variables from parent directory -> ", loaded_env)
load_env()
