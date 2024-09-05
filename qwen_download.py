from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('qwen/Qwen-14B-Chat', cache_dir='/root/autodl-tmp/artboy/base_model/', revision='v1.0.8')

