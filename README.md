# Video Summarization

### Usage

```sh
$ pip install -r requirements.txt
$ python src/main.py test.mp4
```


### Description
In this project, we approach video summarization as a keyframe extraction process. The implemented video summarization system converts input videos into a collection of static images called keyframes, each of which highlighting an event within the video. With keyframe extraction, we are able to provide an abstract representation of the video by a temporal sequence of images. Consequently, the video can be later referred to through its corresponding keyframes in order to facilitate efficient management of video data. To test the quality of the implemented system, the widely
used VSUMM dataset was considered. Results showed a correlation between our outputs and those
provided through a human’s perspective, with up to 70% precision and 80% recall. Initially, the
video is preprocessed and broken down into a less dense form. Secondly, the video is split into
a set of scenes through visual feature analysis, as each scene generally represents a single flow of
events. Finally, keyframes are chosen to summarize the video by clustering each of the extracted
scenes into correlated groups of frames.
### Detailed Report
###### https://drive.google.com/file/d/11gG2L8Yl4YnxPh52mmJk8E2nazy5dTBv/view?usp=sharing
