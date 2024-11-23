from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download


# hf_hub_download(repo_id="haofeixu/depthsplat", filename="config.json")

snapshot_download(repo_id="haofeixu/depthsplat", 
                  cache_dir="/root/autodl-tmp/", 
                  local_dir="/root/autodl-tmp/models")