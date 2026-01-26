from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HUGGINGFACE_TOKEN"))
user = api.whoami()
print(user["name"])
