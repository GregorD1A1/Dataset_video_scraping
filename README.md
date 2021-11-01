# Dataset video scraping
A script that slices a video into frames, recognizes planes in it and creates a dataset of planes based on that. The planes can be easily replaced with any other objects from the coco dataset.

How to use:

1. Clone repository
2. Download to the project folder files for YOLO support:
https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
https://pjreddie.com/media/files/yolov3.weights
3. In project folder create folders 'datase' and 'videos'
4. Put in 'videos' folder some video, that contain planes.
5. Open VScarping.py, in 'video_name' variable write name of your video, also set prefereble interval in variable 'frame_check_interval'. I recomend use interval from 25 to 125 for 25fps video (1s to 5s), in dependance on information density of video.
6. Create project virtual envinronment and install libraryes:

conda install -c conda-forge cv2

pip install tqdm

7. Run VSarping.py. Your raw dataset will be in 'dataset' folder.
