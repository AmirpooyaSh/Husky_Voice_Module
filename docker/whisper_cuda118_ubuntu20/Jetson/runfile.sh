docker run --rm -it \
  --name whisper-jetson-cpu-live \
  --net=host \
  --device /dev/snd \
  --group-add audio \
  -v "$PWD":/workspace \
  whisper-jetson-cpu