docker run -it --rm \
  --name vox_profile \
  --gpus all \
  --net=host \
  --privileged \
  --device /dev/snd:/dev/snd \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/Husky_Voice_Module:/workspace \
  -v vox_hf_cache:/root/.cache/huggingface \
  -v vox_torch_cache:/root/.cache/torch \
  vox-profile-cu102:latest