import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class ChairTableDetector(object):
    def __init__(self):
        PATH_TO_MODEL = '/fs/project/PAS1263/src/models/research/object_detection/chairtable/model_table1/fine_tuned_model/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)



    def get_classification(self, img):
    # Bounding Box Detection.
      with self.detection_graph.as_default():

        # Expand dimension since the model expects image to have shape [1, None, None, 3].

        img_expanded = np.expand_dims(img, axis=0)

        (boxes, scores, classes, num) = self.sess.run(

            [self.d_boxes, self.d_scores, self.d_classes, self.num_d],

            feed_dict={self.image_tensor: img_expanded})

      return boxes, scores, classes, num


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


detector=ChairTableDetector()
testpath='/fs/project/PAS1263/data/ILSVRC/Data/test'
trainpath='/fs/project/PAS1263/data/ILSVRC/Data/train'
for image_path in glob.glob(testpath+'/*.JPEG'):
    path, filename = os.path.split(image_path)
    image = Image.open(image_path,'r')
    if image.mode!='RGB':
       image=image.convert('RGB')

    img=load_image_into_numpy_array(image)
    bndbox=detector.get_classification(img)
    np.savez('./Bndbox/table/'+filename+'.txt', *bndbox)
