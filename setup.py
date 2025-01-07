from setuptools import setup, find_packages

setup(
    name="common_genai_utils",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'huggingface_hub',
        'langchain_anthropic',
        'langchain_ollama',
        'anthropic',
        'langchain_huggingface',
        'chromadb',
        'langchain_ollama',
        'langchain_chroma',
        'langchain',
        'langchain_google_genai',
        'huggingface_hub',
        'nest_asyncio'
    ],  # Add dependencies here
)
