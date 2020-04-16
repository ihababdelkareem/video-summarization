import tkinter as tk
from cv2 import cv2, waitKey
from PIL import ImageTk, Image
import matplotlib
matplotlib.use("TkAgg")


class SceneShow(tk.Frame):
    def __init__(self, parent, scene_list):
        tk.Frame.__init__(self, parent)
        text = tk.Text(self, wrap="none")
        vsb = tk.Scrollbar(orient="vertical", command=text.yview)
        text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)

        for i,scene in enumerate(scene_list):

            arbitraryMiddle = int(scene.getFrameCount()/2)
            cv2.imwrite('images_dir/' + str(i) + '.jpg', scene.getFrameList()[arbitraryMiddle])
            photo = ImageTk.PhotoImage(Image.open('images_dir/' + str(i) + '.jpg').resize((150,100), Image.ANTIALIAS))
            a = tk.Label(self,text=str(i))
            b = tk.Label(self,image=photo)
            c = tk.Label(self,text="CNN",bg="red")
            d = tk.Label(self,text="HSV",bg="green")
            e = tk.Label(self,text="HSV with Next Scene",bg="blue")
            f = tk.Label(self,text="CNN with Next Scene",bg="orange")
            g = tk.Label(self,text="Extract Keyframes (Clustering)",bg="purple")

            def showCustomScene(event,scene):
                scene.playSceneWithParentIndices()

            def showPlot(event,scene,method,params):
                scene.multiplePlots([method],[params])

            def concat_plot(event,scene,method,params):
                # print(scene.nextScene.getFrameList())
                scene.trimmedPlotWithNextScene(scene.nextScene,method,params)

            def clusterScene(event,scene,method,params):
                scene.extractKeyFrames(method,params)

            b.bind("<Button-1>", lambda event,scene=scene:showCustomScene(event = event, scene=scene))
            c.bind("<Button-1>", lambda event,scene=scene:showPlot(event = event, scene=scene,method = 'cnn',params= {'model': 'keras'}))
            d.bind("<Button-1>", lambda event,scene=scene:showPlot(event = event, scene=scene,method = 'color',params= {'difference_metric':'correlation'}))
            e.bind("<Button-1>", lambda event,scene=scene:concat_plot(event = event, scene=scene,method = 'color',params= {'difference_metric':'correlation'}))
            f.bind("<Button-1>", lambda event,scene=scene:concat_plot(event = event, scene=scene,method = 'cnn',params= {'model': 'keras'}))
            g.bind("<Button-1>", lambda event,scene=scene:clusterScene(event = event, scene=scene,method = 'cnn',params= {'model': 'keras'}))

            b.image = photo  # keep a reference
            text.window_create("end", window=a)
            text.window_create("end", window=b)
            text.window_create("end", window=c)
            text.window_create("end", window=d)
            text.window_create("end", window=e)
            text.window_create("end", window=f)
            text.window_create("end", window=g)
            text.insert("end", "\n")

        text.configure(state="disabled")
