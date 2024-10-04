import os
from gpt_researcher.utils import get_deployment_and_api_key_openai

OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL","text-embedding-3-small-1")


class Memory:
    def __init__(self, embedding_provider, headers=None, **kwargs):
        _embeddings = None
        headers = headers or {}
        match embedding_provider:
            case "ollama":
                from langchain_community.embeddings import OllamaEmbeddings

                _embeddings = OllamaEmbeddings(
                    model=os.environ["OLLAMA_EMBEDDING_MODEL"],
                    base_url=os.environ["OLLAMA_BASE_URL"],
                )
            case "custom":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=os.environ.get("OPENAI_EMBEDDING_MODEL", "custom"),
                    openai_api_key=headers.get(
                        "openai_api_key", os.environ.get("OPENAI_API_KEY", "custom")
                    ),
                    openai_api_base=os.environ.get(
                        "OPENAI_BASE_URL", "http://localhost:1234/v1"
                    ),  # default for lmstudio
                    check_embedding_ctx_length=False,
                )  # quick fix for lmstudio
            case "openai":
                from langchain_openai import AzureOpenAIEmbeddings

                model_name = OPENAI_EMBEDDING_MODEL

                azure_endpoint, api_key = get_deployment_and_api_key_openai(model_name)

                _embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    model=model_name,
                    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
                )
            case "azure_openai":
                from langchain_openai import AzureOpenAIEmbeddings

                _embeddings = AzureOpenAIEmbeddings(
                    deployment=os.environ["AZURE_EMBEDDING_MODEL"], chunk_size=16
                )
            case "huggingface":
                from langchain.embeddings import HuggingFaceEmbeddings

                # Specifying the Hugging Face embedding model all-MiniLM-L6-v2
                _embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

            case _:
                raise Exception("Embedding provider not found.")

        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings
