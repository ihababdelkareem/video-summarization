import sys
import cv2
import copy
import os
sys.path.append(os.path.dirname(__file__).replace('src','',1))
from scipy import spatial
import src.utils.inputVideo
import src.utils.imageHistogram as histUtility
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import src.utils.localFeatures as local
from src.utils.keyframe import KeyFrame
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import distance_metric
from pyclustering.utils import calculate_distance_matrix

class Scene(src.utils.inputVideo.InputVideo):
    #Has all Input video functionality , scene specific functionality to be added
    def __init__(self,path,starting_index,ending_index,diff_list_dict,scene_id,keras_model,dr_model,scene_kf_path,config):
        super().__init__(path,config)
        """
        Starting Index : Starting frame index relative to the original Video
        Ending Index : Ending frame index relative to the original Video
        Length of scene frame list = (ending_index - starting_index +1)
        Diff List of the scene = video_diff_list[starting_index:ending_index] (ending_index not included)
        explained -> original_diff_list [starting_index] = diff(original_frame_list of starting_index+1 and starting_index)
        and          original_diff_list [ending_index] = diff(original_frame_list of ending_index and ending_index -1)

        """
        self.starting_index = starting_index
        self.ending_index   = ending_index
        self.diff_list_dict = copy.deepcopy(diff_list_dict) # no change done on the original dict
        self.nextScene = None
        self.scene_id = scene_id
        self.keras_model = keras_model
        self.dr_model = dr_model
        self.scene_kf_path = scene_kf_path
        for key,value in self.diff_list_dict.items():
            diff_list,feat_list = value # From Parent Video
            scene_diff_list,scene_feat_list = diff_list[starting_index:ending_index],feat_list[starting_index:ending_index+1]
            self.diff_list_dict[key] = scene_diff_list,scene_feat_list # internal diff list of length (scene frame list - 1 )
        print('Scene: {} to {}'.format(starting_index,ending_index))
    def playSceneWithParentIndices(self):
        for i,frame in enumerate(self.getFrameList()):
            cv2.imshow("Frame",frame)
            k = cv2.waitKey(0) & 0xff
            if k == 27:
                break
            print(self.starting_index+i)


    def trimmedPlotWithNextScene(self,next_scene,method,params):
        self_list = self.getFrameList()
        next_scene_list = next_scene.getFrameList()
        self_list=self_list[0:int(0.75*len(self_list))]
        next_scene_list = next_scene_list[int(0.25*len(next_scene_list)):]
        concat_list = self_list+next_scene_list
        self.multiplePlots(method_list=[method],param_list=[params],external_list=concat_list,interactiveHover =False)

    def clusterSceneKMedoid(self,method,params,show_frames_per_cluster=False):
        fc  = len(self.getFrameList())
        print("FC {}".format(fc))
        if fc > 3:
            sample = self.getAdjacentDifferenceList(method,params,getFeatures=True)
            initial_medoids = [1, 1, 1]
            kmedoids_instance = kmedoids(sample, initial_medoids)
            kmedoids_instance.process()
            clusters = kmedoids_instance.get_clusters()
            medoids = kmedoids_instance.get_medoids()
        else :
            medoids = list(range(fc))
        print("Medoids: {}".format(medoids))
        keyframe_obj_list = []
        for medoid_index in medoids:
            kf = KeyFrame(image=self.getFrameList()[medoid_index],scene_id=self.scene_id,index=self.starting_index+medoid_index)
            keyframe_obj_list.append(kf)
        if(self.config['scene_based_removal']):
            print('Internal Scene Selection')
            feature_vectors = self.getAdjacentDifferenceList(method, params, getFeatures=True)
            feature_vectors = [feature_vectors[kf.index-self.starting_index] for kf in keyframe_obj_list]
            final_kfs = self.finalKeyframeDuplicateRemoval(keyframe_obj_list,self.config['scene_based_removal_thresh'],feature_vectors)
        else:
            final_kfs = keyframe_obj_list
        for i, kf in enumerate(final_kfs):
            cv2.imwrite('{}/{}.jpg'.format(self.scene_kf_path, kf.index), kf.image)
        return final_kfs

        return keyframe_obj_list

        """
        As the first step for inner scene keyframe extraction , the scene is clustered into k clusters
        where the optimal value of k is decided by the silhouette measure , a list mapping clusters to
        frame indices is returned
        """
    def clusterScene(self,method,params,show_frames_per_cluster=False):
        K = range(2, 4)
        feature_vectors = self.getAdjacentDifferenceList(method,params,getFeatures=True)
        silhouette_scores = []
        k_means_models = []

        #Try all values of k in the range(2,5)
        for k in K:
         kmeans_model = KMeans(n_clusters=k,max_iter=500)
         kmeans_model.fit(feature_vectors)
         k_means_models.append(kmeans_model)
         labels=kmeans_model.labels_
         #Compute silhouette_score at each k
         silhouette_scores.append(silhouette_score(feature_vectors, labels, metric = 'euclidean'))

        # print(silhouette_scores)
        optimal_cluster_count = 2+silhouette_scores.index(max(silhouette_scores))
        print('Optimal Cluster Count: {}'.format(optimal_cluster_count))
        print('Frame to Cluster Mapping: ')
        # print(k_means_models[optimal_cluster_count-2].predict(feature_vectors))

        cluster_to_frame_mapping = []
        for i in range(optimal_cluster_count):
            cluster_to_frame_mapping.append([])
        for (frame_index,cluster_index) in enumerate(k_means_models[optimal_cluster_count-2].predict(feature_vectors)):
            cluster_to_frame_mapping[cluster_index].append(frame_index)

        #Show frames for each cluster

        if(show_frames_per_cluster):
            for cluster_index,cluster in enumerate(cluster_to_frame_mapping):
                for img_index in cluster:
                    cv2.imshow('Cluster: {},Frame: {}'.format(cluster_index,img_index),self.getFrameList()[img_index])
                    cv2.waitKey()
                    cv2.destroyAllWindows()


        #Return list of clusters and corresponding Frame Indices
        return cluster_to_frame_mapping


    """
    For each resultant cluster and its corresponding frames , we chose the representative keyframe
    as the frame which has the highest correlation with the remaining frames (HSV)
    """
    def chooseKeyframes(self,cluster_to_frame_mapping,method,params):
        # print(cluster_to_frame_mapping)
        feature_vectors = self.getAdjacentDifferenceList(method,params,getFeatures=True)
        optimal_cluster_count = len(cluster_to_frame_mapping)
        cluster_to_medoid = []
        for i in range(optimal_cluster_count):
            cluster_to_medoid.append(0)

        for cluster_index,cluster in enumerate(cluster_to_frame_mapping):
            max_index_of_sum = 0
            max_sum = 0
            for i in range(len(cluster)):
                current_sum = 0
                for j in range(len(cluster)):
                    fi = feature_vectors[cluster[i]]
                    fj = feature_vectors[cluster[j]]
                    current_sum+=self.cossim(fi,fj)
                if (current_sum>max_sum):
                    max_sum = current_sum
                    max_index_of_sum = cluster[i]
            cluster_to_medoid[cluster_index] = max_index_of_sum # maps cluster to its medoid frame index (relative to the scene)

        print(cluster_to_medoid)

        return cluster_to_medoid

    def extractKeyFramesByKMeans(self,method,params):
        print('Clustering Scene')
        cluster_to_frame_mapping = self.clusterScene(method,params) if len(self.getFrameList())>5 else [list(range(len(self.getFrameList())))]
        print('Selecting Keyframes from Each Cluster')
        cluster_to_keyframe_mapping = self.chooseKeyframes(cluster_to_frame_mapping,method,params)
        keyframe_obj_list = []
        for medoid_index in cluster_to_keyframe_mapping:
             kf = KeyFrame(image=self.getFrameList()[medoid_index],scene_id=self.scene_id,index=self.starting_index+medoid_index)
             keyframe_obj_list.append(kf)

        if(self.config['scene_based_removal']):
            print('Internal Scene Selection')
            feature_vectors = self.getAdjacentDifferenceList(method, params, getFeatures=True)
            feature_vectors = [feature_vectors[kf.index-self.starting_index] for kf in keyframe_obj_list]
            final_kfs = self.finalKeyframeDuplicateRemoval(keyframe_obj_list,self.config['scene_based_removal_thresh'],feature_vectors)
            print('{} -> {}'.format(len(keyframe_obj_list),len(final_kfs)))
        else:
            final_kfs = keyframe_obj_list
        for i, kf in enumerate(final_kfs):
            cv2.imwrite('{}/{}.jpg'.format(self.scene_kf_path, kf.index), kf.image)
        return final_kfs
