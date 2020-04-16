import cv2 as cv
def writeVideoToPath(frame_list,path,fps,w,h):
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    output_video = cv.VideoWriter(path,fourcc, fps, (int(w),int(h)))
    for frame in frame_list:
        output_video.write(frame)
    output_video.release()
