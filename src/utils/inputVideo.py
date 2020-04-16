import sys
import os
sys.path.append(os.path.dirname(__file__).replace('src', '', 1))
import src.utils.smooth as smooth
from src.utils.bow.vbow import BOV
import src.utils.localFeatures as local
import src.utils.outputVideo as outputVideo
import src.utils.imageHistogram as histUtility
import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
import math
import numpy as np
import shutil
import time
import multiprocessing

class InputVideo:

    def __init__(self,path,config,resize = False):
        self.path = path
        self.video = cv.VideoCapture(path)
        if(self.video.isOpened() == False):
            raise Exception('Error while opening video {}'.format(path))
        else:
            self.FRAME_COUNT = self.video.get(cv.CAP_PROP_FRAME_COUNT)
            self.FRAME_RATE = self.video.get(cv.CAP_PROP_FPS)
            self.FRAME_WIDTH = self.video.get(cv.CAP_PROP_FRAME_WIDTH)
            self.FRAME_HEIGHT = self.video.get(cv.CAP_PROP_FRAME_HEIGHT)
            self.frame_list = []
            self.keyframe_list = []
            self.diff_list_dict = {}  # Maps Params to diff list , calculate diff list per param only once
            self.feat_vect_dict = {}  # Maps Params to feature_vect list , calculate features only once
            self.vbow = None
            self.keras_model = None
            self.dr_model = None
            self.i2v = None
            self.summarization_data_path = self.path[0:self.path.index('.')]
            self.config = config
            self.resize = resize
            self.getFrameList()
            if os.path.exists(self.summarization_data_path):
                shutil.rmtree(self.summarization_data_path)  # Remove the old summarization data storage

    def reInit(self):
        self.video.release()
        self.video = cv.VideoCapture(self.path)

    def getVideoName(self):
        return self.path.split('/')[-1]

    def getFrameList(self):
        if(len(self.frame_list) == 0):
            t1 = time.time()
            ret, frame = self.video.read()
            i = 0
            while ret:
                i += 1
                self.frame_list.append(frame)
                ret, frame = self.video.read()
            print('{} frames from {} read in {} seconds'.format(len(self.frame_list),self.getVideoName(),round(time.time() - t1,2)))
        self.setNextFrameIndex(0)
        return self.frame_list

    def getFrameRate(self):
        return self.FRAME_RATE

    def getFrameCount(self):
        return self.FRAME_COUNT

    def getFrameWidth(self):
        return self.FRAME_WIDTH

    def getFrameHeight(self):
        return self.FRAME_HEIGHT

    def getLengthInSeconds(self):
        return int(self.FRAME_COUNT / self.FRAME_RATE)

    def getFormattedVideoLenghtInSeconds(self):
        seconds = self.getLengthInSeconds()
        minutes = int(seconds / 60)
        seconds = seconds % 60
        return str(minutes) + ':' + (str(seconds) if len(str(seconds)) == 2 else "0" + str(seconds))

    def getNthFrame(self, n):
        # sets index of next frame to n and returns it
        self.setNextFrameIndex(n)
        frame = self.__next__()  # might raise "Frame Count Exceeded"
        self.setNextFrameIndex(0)
        return frame

    def setNextFrameIndex(self, n):
        # set index of frame to be retrieved by __next__
        self.video.set(cv.CAP_PROP_POS_FRAMES, n)

    def getNextFrameIndex(self):
        return self.video.get(cv.CAP_PROP_POS_FRAMES)

    def getFrameAtSecond(self, sec):
        self.video.set(cv.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = self.video.read()
        self.setNextFrameIndex(0)  # return the pointer to defualt in case we use __next__
        return hasFrames, image

    def getSampledFrameList(self, new_fps):
        print("##################")
        t1 = time.time()
        print("Sampling to {} FPS".format(new_fps))
        skip = int(self.FRAME_RATE / new_fps)
        print('{}/{} -> {}'.format(self.FRAME_RATE,new_fps,skip))
        frame_list = self.getFrameList()[::skip] if skip>=1 else self.getFrameList()[::1]
        if(self.resize):
            print("Resizing to {}x{}".format(self.config['width'],self.config['height']))
            frame_list = [cv.resize(i,(self.config['width'],self.config['height'])) for i in frame_list]
        print("Done in {} Sec".format(round(time.time()-t1)))
        return frame_list

    def play(self):
        # Use unified List
        # plays video till frames end
        self.setNextFrameIndex(0)  # play from the beginning
        for frame in self.getFrameList():
            cv.imshow('Frame', frame)
            if (cv.waitKey(25) & 0xFF == ord('q')):  # press q to stop playback
                break
        self.setNextFrameIndex(0)  # return to the beginning
        cv.destroyAllWindows()

    def initBagOfWords(self, clusters, training_list):
        self.vbow = BOV(no_clusters=clusters, images=training_list)

    def multiplePlots(self, method_list, param_list, interactiveHover=True, save_as_path=None, drawThres=False, smooth_plot=False, external_list=None):
        # Use unified List
        # Make modualar + support multiple plots based on input
        # self.reInit()
        frame_list = self.getFrameList() if external_list == None else external_list
        # get array of difference arrays
        if external_list == None:
            differences = [self.getAdjacentDifferenceList(
                method_list[i], param_list[i], False) for i in range(len(method_list))]
        else:
            differences = [self.getAdjacentDifferenceList(
                method_list[i], param_list[i], False, external_list) for i in range(len(method_list))]

        # if(drawThres and len(method_list)>1):
        #     raise Error("Use only one method when drawing the threshold")

        """
        Callback for Hover
        """
        def onHover(event, l):  # event is the location of the mouse
            cv.destroyAllWindows()
            x = int(math.ceil(event.xdata)) if event.xdata != None else None
            if(l.contains(event)[0]):
                cv.imshow('Frame: ' + str(x), frame_list[x])
        ################################################

        """
        Regular and Smoothed Plots
        """
        fig, ax = plt.subplots(2) if smooth_plot else plt.subplots(1)
        original_ax = None
        for diff_list in differences:
            original_ax = ax[0] if smooth_plot else ax
            original_scatter = original_ax.scatter(list(range(len(diff_list))), diff_list, s=10)
            original_ax.plot(diff_list, markersize=5)  # plot original curve (above)
            if(smooth_plot):
                smoothed_plot = smooth.smooth(np.array(diff_list))
                smoothed_scatter = ax[1].scatter(list(range(len(smoothed_plot))), smoothed_plot, s=10)
                ax[1].plot(smoothed_plot, markersize=5)  # plot smoothed curve (below)

        original_ax.legend(method_list, loc='upper left')
        original_ax.set_title("Before Smoothing")
        if(smooth_plot):
            ax[1].legend(method_list, loc='upper left')
            ax[1].set_title("After Smoothing")

        if(interactiveHover):
            fig.canvas.mpl_connect('motion_notify_event', lambda event: onHover(event, original_scatter))

        if(drawThres):  # IMBORTUNT
            for diff_list in differences:
                thres1, thres2 = self.__getCutThreshold(diff_list)
                original_ax.plot([0, len(diff_list)], [thres1, thres1])
                original_ax.plot([0, len(diff_list)], [thres2, thres2])

        plt.show()
        if save_as_path != None:
            fig.set_size_inches((15, 8), forward=False)
            video_name = self.path.split('.')[0].split('/')[-1]
            fig.savefig('{}/{}.png'.format(save_as_path, video_name), dpi=400)
        self.setNextFrameIndex(0)

    def getSampledInputVideo(self, fps):
        frame_list = self.getSampledFrameList(fps)
        name = '{}_{}.{}'.format(self.path.split('.')[0], fps, self.path.split('.')[1])
        width = self.config['width'] if self.resize else self.FRAME_WIDTH
        height = self.config['height'] if self.resize else self.FRAME_HEIGHT
        outputVideo.writeVideoToPath(frame_list, name, fps, width,height)
        return InputVideo(name,self.config)

    def getPairwiseFrameTupleList(self, external_list=None):
        # Use unified List
        frame_list = self.getFrameList() if external_list == None else external_list
        return [(frame_list[i], frame_list[i + 1]) for i in range(len(frame_list) - 1)]

    def getAdjacentDifferenceList(self, method, params, showAndWait=False, external_list=None, getFeatures=False, loadCNNfromCache=False):
        # Use unified List
        frame_list = self.getFrameList() if external_list == None else external_list
        pairwise_tuple_list = self.getPairwiseFrameTupleList() if external_list == None else self.getPairwiseFrameTupleList(external_list)
        # self.dr_model = DeepRankingModel() if self.dr_model == None else self.dr_model

        # check if I calculated this feat list/diff list previously
        if '{}{}'.format(method, str(params)) in self.diff_list_dict.keys() and external_list == None:
            print("{} found in dict".format(method))
            diff_list, feat_list = self.diff_list_dict.get('{}{}'.format(method, str(params)))
            return feat_list if getFeatures else diff_list

        def getPreTrainedDiff(params):
            model = params['model']

            if(model == 'pytorch-i2v'):
                if self.i2v == None:
                    from src.utils.img2vec_pytorch.img_to_vec import Img2Vec
                    self.i2v = Img2Vec(model='alexnet')
                feat_list = [i2v.get_vec(img) for img in frame_list]
                diff_list = [i2v.getPairDifference(feat_list[i], feat_list[i + 1]) for i in range(len(feat_list) - 1)]
                return diff_list, feat_list

            elif (model == 'keras'):
                if self.keras_model == None and loadCNNfromCache == False:
                    from src.utils.keras_pretrained.keras_ft import KerasModel
                    self.keras_model = KerasModel()

                feat_list = self.loadCNNfromCache() if loadCNNfromCache == True else [
                    self.keras_model.getFeatureVector(img) for img in frame_list]
                diff_list = [self.cossim(feat_list[i], feat_list[i + 1]) for i in range(len(feat_list) - 1)]
                return diff_list, feat_list

            elif (model == 'dr'):
                if self.dr_model == None:
                    from src.utils.deep_ranking.deep_ranking_model import DeepRankingModel
                    self.dr_model = DeepRankingModel()
                feat_list = dr_model.getFeatureVectorList(frame_list)
                diff_list = [dr_model.getPairDifference(feat_list[i], feat_list[i + 1])
                             for i in range(len(feat_list) - 1)]

                return diff_list, feat_list

        def getSiftDiff(params):
            descriptor = params['descriptor']
            feat_list = local.getFeatureVectorList(imgList, descriptor)
            diff_list = []
            for i in range(len(feat_list) - 1):
                kps1, des1 = feat_list[i]
                kps2, des2 = feat_list[i + 1]
                diff_list.append(local.getAdjacentMatch(kps1, des1, kps2, des2, descriptor))
            return diff_list, feat_list

        def getHSVDiff(params):
            difference_metric = params['difference_metric']
            feat_list = histUtility.hsvHistsForSeriesOfImages(frame_list)
            diff_list = histUtility.adjacentHistComparison(feat_list, difference_metric)
            return diff_list, feat_list

        def getVBOWDiff(params):
            clusters = params['clusters']
            # training_fps = params['training_fps'] # make codebook from video sampled at 1 fps for example
            # difference_metric = params['difference_metric']
            # training_video = self.getSampledInputVideo(training_fps)
            self.initBagOfWords(clusters, frame_list)
            print('Training VBOW Codebook')
            self.vbow.trainModel()

            feat_list = [self.vbow.getImageVocab(i) for i in frame_list]
            diff_list = [self.cossim(feat_list[i], feat_list[i + 1]) for i in range(len(feat_list) - 1)]
            return diff_list, feat_list

        def showFrameAndWaitOnKey(frame, i):
            cv.imshow("Frame: {}".format(i), frame)
            cv.waitKey()
            cv.destroyAllWindows()

        returned_val = {
            'local': getSiftDiff,
            'color': getHSVDiff,
            'vbow': getVBOWDiff,
            'cnn': getPreTrainedDiff
        }

        returned_diff_list, returned_feature_list = returned_val[method](params)

        if showAndWait:
            for i in range(len(frame_list)):
                if(i != 0):
                    print(returned_list[i - 1])
                showFrameAndWaitOnKey(frame_list[i], i)

        if external_list == None:
            self.diff_list_dict['{}{}'.format(method, str(params))] = returned_diff_list, returned_feature_list

        return returned_feature_list if getFeatures else returned_diff_list

    """
    Scene Cutting Methodology
    """

    def __getCutThreshold(self, diff_list):  # we may change this to get bigger scenes
        import statistics
        if(self.config['scene_cut_thresh']=='auto'):
            discrete = list(set(int(x * 100) for x in diff_list))  # A list of the set of discretized differences
            discrete.sort()
            return statistics.median(discrete)/100,statistics.median(discrete)/100,
            discrete2 = discrete[:len(discrete) * 3 // 4]
            return 0.01 * sum(discrete) / len(discrete) if len(discrete) != 0 else 0, 0.01 * sum(discrete2) / len(discrete2) if len(discrete2) != 0 else 0
        else:
            return self.config['scene_cut_thresh'],self.config['scene_cut_thresh']
    def getSceneBoundariesFromThreshCut(self, method, params, disolve_window_duration):
        disolve_window_frame_limit = self.getFrameRate() * disolve_window_duration
        diff_list = self.getAdjacentDifferenceList(method, params)
        thresh1, thresh2 = self.__getCutThreshold(diff_list)
        list_below = [0] + [i for (i, val) in enumerate(diff_list) if val < thresh2] + [len(diff_list) - 1]
        scene_pair_list = []
        for i, val in enumerate(list_below[:-1]):
            if(list_below[i + 1] - val > disolve_window_frame_limit):
                scene_pair_list.append((val, list_below[i + 1]))

        return scene_pair_list

    def writeAndGetScenes(self, scene_boundaries_list):
        import src.utils.scene as scene
        scenes_list = []
        scenes_folder_path = self.summarization_data_path + '/scenes'
        os.makedirs(scenes_folder_path)
        kfs_per_scene_path = self.summarization_data_path + '/kfs_per_scene'
        os.makedirs(kfs_per_scene_path)

        for i, scene_boundaries in enumerate(scene_boundaries_list):
            scene_path = '{}/{}.{}'.format(scenes_folder_path, i, self.path.split('.')[1])
            scene_kf_path = '{}/{}'.format(kfs_per_scene_path,i)
            os.makedirs(scene_kf_path)
            outputVideo.writeVideoToPath(self.getFrameList(
            )[scene_boundaries[0] + 1:scene_boundaries[1] + 1], scene_path, self.getFrameRate(), self.FRAME_WIDTH, self.FRAME_HEIGHT)
            scenes_list.append(scene.Scene(scene_path, starting_index=scene_boundaries[0] + 1, ending_index=scene_boundaries[1],
                                           diff_list_dict=self.diff_list_dict, scene_id=i, keras_model=self.keras_model,
                                            dr_model=self.dr_model,scene_kf_path=scene_kf_path,config=self.config))

        for i, scene in enumerate(scenes_list[:-1]):
            scenes_list[i].nextScene = scenes_list[i + 1]
        return scenes_list

    def showScenes(self, scene_list):
        root = tk.Tk()
        SceneShow(root, scene_list).pack(fill="both", expand=True)
        root.mainloop()

    def extractKeyframesFromScene(self, scene, proc_num=None, return_dict=None):
        if(self.config['clustering']=='kmeans'):
            kfs = scene.extractKeyFramesByKMeans(method=self.config['scene_processing_features'], params=self.config['scene_processing_features_params'])
        else:
            kfs = scene.clusterSceneKMedoid(method='cnn', params={'model': 'keras'})
        if(proc_num != None):
            return_dict[proc_num] = kfs
        else:
            return kfs

    def generateKeyframes_multiprocessing(self):
        scene_list = self.writeAndGetScenes(self.getSceneBoundariesFromThreshCut(
            self.config['scene_cut_features'],self.config['scene_cut_features_params'], self.config['min_scene_length']))
        processes = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for i, scene in enumerate(scene_list):
            p = multiprocessing.Process(target=self.extractKeyframesFromScene, args=(scene, i, return_dict))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()
        kfs = []
        for k, v in sorted(return_dict.items()):
            kfs += v

        before_path = self.summarization_data_path + '/kfs_before'
        os.makedirs(before_path)
        for i, kf in enumerate(kfs):
            cv.imwrite('{}/{}.jpg'.format(before_path, i), kf.image)

        if(self.config['global_removal']):
            feature_vectors = self.getAdjacentDifferenceList('cnn', self.config['cnn_params'], getFeatures=True)
            feature_vectors = [feature_vectors[kf.index] for kf in kfs]
            final_kfs = self.finalKeyframeDuplicateRemoval(kfs, self.config['global_removal_thresh'],feature_vectors)
            after_path = self.summarization_data_path + '/kfs_after'
            os.makedirs(after_path)
            for i, kf in enumerate(final_kfs):
                cv.imwrite('{}/{}.jpg'.format(after_path, i), kf.image)
            return final_kfs
        else:
            return kfs

    def generateKeyframes_sequential(self):
        kfs = []
        scene_list = self.writeAndGetScenes(self.getSceneBoundariesFromThreshCut(
            self.config['scene_cut_features'],self.config['scene_cut_features_params'], self.config['min_scene_length']))
        for scene in scene_list:
            kfs += self.extractKeyframesFromScene(scene)

        before_path = self.summarization_data_path + '/kfs_before'
        os.makedirs(before_path)
        for i, kf in enumerate(kfs):
            cv.imwrite('{}/{}.jpg'.format(before_path, i), kf.image)

        if(self.config['global_removal']):
            feature_vectors = self.getAdjacentDifferenceList('cnn', self.config['cnn_params'], getFeatures=True)
            feature_vectors = [feature_vectors[kf.index] for kf in kfs]
            final_kfs = self.finalKeyframeDuplicateRemoval(kfs, self.config['global_removal_thresh'],feature_vectors)
            after_path = self.summarization_data_path + '/kfs_after'
            os.makedirs(after_path)
            for i, kf in enumerate(final_kfs):
                cv.imwrite('{}/{}.jpg'.format(after_path, i), kf.image)
            return final_kfs
        else:
            return kfs

    def finalKeyframeDuplicateRemoval(self, kfs, thresh,feature_vectors):
        def getConcatedImage(img1, img2):
            numpy_horizontal = np.hstack((img1, img2))
            numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)
            cv.imshow('x', numpy_horizontal_concat)
            cv.waitKey()
            cv.destroyAllWindows()

        final_set = []

        kf_with_vec = zip(kfs, feature_vectors)

        def already_exists(kf_check, final_set):
            for kf_tuple in final_set:
                diff_1_2_cnn = self.cossim(kf_check[1], kf_tuple[1])
                feat_list = histUtility.hsvHistsForSeriesOfImages([kf_check[0].image,kf_tuple[0].image])
                diff_1_2_hsv = histUtility.adjacentHistComparison(feat_list,'correlation')[0]
                if(diff_1_2_cnn > thresh or diff_1_2_hsv>self.config['global_hsv_thresh']):
                    return True
            return False
        for kf_tuple in kf_with_vec:
            if(not already_exists(kf_tuple, final_set)):
                final_set.append(kf_tuple)

        final_set = [kf[0] for kf in final_set]
        return final_set

    def cacheCNN(self):
        path = '{}/{}'.format(self.config['cnn_vects_path'],self.getVideoName()[:self.getVideoName().find('.')])
        cnn_vects = self.getAdjacentDifferenceList(method='cnn', params={'model': 'keras'}, getFeatures=True)
        np.save(path, np.array(cnn_vects))

    def loadCNNfromCache(self):
        path = '{}/{}.npy'.format(self.config['cnn_vects_path'],self.getVideoName()[:self.getVideoName().find('.')])
        np_array = np.load(path, allow_pickle=True)
        ret_list = [np_array[i] for i in range(np_array.shape[0])]
        return ret_list

    def cossim(self, vec1, vec2):
        from scipy import spatial
        return 0.5 * (2 - spatial.distance.cosine(vec1, vec2))  # cosine sim between a and b

    def storeToFirebase(self):
        from utils.firebase.firebaseService import FirebaseService
        fb = FirebaseService()
        fb.storeVideo(self)
