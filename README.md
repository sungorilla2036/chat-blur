# chat-blur

## Prerequisites
Python 3.7 or 3.8

FFMPEG (for most video codecs)

paddlepaddle or paddlepaddle-gpu See https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html

## Install requirements.txt
`pip install -r requirements.txt`

## Usage
### Example

`python blurtext.py input_video.mp4 --use_gpu --bounding_rect 0 0 1920 1080 --output_path output_video.mp4`
