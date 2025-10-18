import httpx
import openai

MODEL_VISION = "gpt-5"

def openai_client() -> OpenAI:
    # Client HTTPx personnalisable :
    http_client = httpx.Client(
        timeout=500.0,  # allonge franchement le timeout
        # http2=False,            # désactive HTTP/2 (certains proxies le cassent sur POST)
        verify=True,  # laisse True (si besoin, tu peux pointer vers un bundle certifi)
    )
    return OpenAI()
    #     timeout=500.0,
    #     max_retries=5,
    #     #http_client=http_client,
    # )
