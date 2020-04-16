class KeyFrame:
    def __init__(self,image,scene_id,index):
        self.image = image
        self.scene_id = scene_id
        self.index = index # Index relative to the whole video
        self.google_cloud_data = self.generateCloudData() # stores labels,web_entities,tags,object,etc. from google cloud vision

    def generateCloudData(self):
        """
        To-Do:
        Develop a dictionary to store all the generated data for this keyframe from gcp
        example :
        keyframe_data = {'labels'=['landscape','nature',...],'web_entities'=['meme','9gag',...],'object'=['man','tree',...]}
        """
        keyframe_data = {}
        mock_keyframe_data = {
        'labels': ['1','2','3','4'],
        'web_entities': ['a','b','c','d']
        }
        return mock_keyframe_data

    def getStorageObject(self):
        return {'scene_id':self.scene_id,'index':self.index,'cloud_data':self.google_cloud_data}
