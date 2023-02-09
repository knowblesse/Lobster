
from pathlib import Path
import cv2 as cv
from tkinter.filedialog import askdirectory
import numpy as np

# TODO : Only Mark Nestzone Data
# Constants
TANK_PATH = Path(askdirectory())

# Find the path to the video
vidlist = []
vidlist.extend([i for i in TANK_PATH.glob('*.mkv')])
vidlist.extend([i for i in TANK_PATH.glob('*.avi')])
vidlist.extend([i for i in TANK_PATH.glob('*.mp4')])
vidlist.extend([i for i in TANK_PATH.glob('*.mpg')])
if len(vidlist) == 0:
    raise(BaseException(f'ReLabeler : Can not find video in {TANK_PATH}'))
elif len(vidlist) > 1:
    raise(BaseException(f'ReLabeler : Multiple video files found in {TANK_PATH}'))
else:
    path_video = vidlist[0]

# Load the video and the label data
vid = cv.VideoCapture(str(path_video))
num_frame = vid.get(cv.CAP_PROP_FRAME_COUNT)
fps = vid.get(cv.CAP_PROP_FPS)

# Find prev dataset if exist
if [path for path in TANK_PATH.glob('data_distracted.csv')]:
    data_distracted = np.loadtxt(next(TANK_PATH.glob('data_distracted.csv')), delimiter=',')
else:
    data_distracted = np.zeros(int(num_frame))

# Main UI functions and callbacks
def getFrame(current_frame):
    vid.set(cv.CAP_PROP_POS_FRAMES, current_frame)
    ret, image = vid.read()
    
    if not ret:
        raise(BaseException('Can not read the frame'))
    cv.putText(image, f'{current_frame} - {current_frame/num_frame*100:.2f}% - Distracted:{convertToggle2Bool()[current_frame]} - {data_distracted[current_frame]}', [0,int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)-10)],fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=[255,255,255], thickness=1)
    cv.putText(image, 'a/f : +-1 min | s/d : +-1 label | e : error frame | w : excursion | q : quit', [0,15], fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=[255,255,255], thickness=1)
    return image

def convertToggle2Bool():
    outputBool = np.zeros(int(num_frame), dtype=bool)
    for i in range(np.sum(data_distracted == 2)):
        outputBool[np.where(data_distracted == 1)[0][i] : np.where(data_distracted == 2)[0][i]] = True
    if np.sum(data_distracted == 1) > np.sum(data_distracted == 2):
        outputBool[np.where(data_distracted == 1)[0][-1]:] = True
    return outputBool

# Start Main UI
key = ''
cv.namedWindow('Main')
current_frame = 0
while key!=ord('q'):
    cv.imshow('Main', getFrame(current_frame))
    key = cv.waitKey()
    if key == ord('a'): # backward 5 frame
        current_frame = int(np.max([0, current_frame - 5]))
    elif key == ord('f'): # forward 5 frame
        current_frame = int(np.min([num_frame-1, current_frame + 5]))
    elif key == ord('s'): # backward 1 label
        current_frame = int(np.max([0, current_frame - 1]))
    elif key == ord('d'): # forward 1 label
        current_frame = int(np.min([num_frame-1, current_frame + 1]))
    elif key == ord('j'): # mark start behav
        if data_distracted[current_frame] != 1:
            data_distracted[current_frame] = 1
        else:
            data_distracted[current_frame] = 0
    elif key == ord('k'): # mark end behav
        if data_distracted[current_frame] != 2:
            data_distracted[current_frame] = 2
        else:
            data_distracted[current_frame] = 0
    elif key == ord('h'): # Go to the last mark:
        current_frame = int(np.where(data_distracted > 0)[0][-1])

cv.destroyWindow('Main')
np.savetxt(TANK_PATH / 'data_distracted.csv', data_distracted,fmt='%d',delimiter=',')
np.savetxt(TANK_PATH / 'bool_distracted.csv', convertToggle2Bool(),fmt='%d',delimiter=',')
