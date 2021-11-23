import time
import yaml
import wget
from utils import *
from base_models import AlexNet, C3DNet, convert_to_fcn, C3DNet2, SIMPLE_CNN
from base_models import I3DNet
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.layers import GRU, LSTM, GRUCell
from tensorflow.keras.layers import Dropout, LSTMCell, RNN
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Average, Add
from tensorflow.keras.layers import ConvLSTM2D, Conv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import vgg19, resnet, efficientnet, mobilenet_v2
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, dot, concatenate, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from vit_keras import vit
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from keras.layers.core import Reshape

## For deeplabV3 (segmentation)
from PIL import Image
import tensorflow as tf
import tarfile
import os
import cv2

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np

from wandb.keras import WandbCallback


###############################################
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    # colormap = np.zeros((256, 3), dtype=np.uint8)
    # colormap[0] = [128, 64, 128]
    # colormap[1] = [244, 35, 232]
    # colormap[2] = [70, 70, 70]
    # colormap[3] = [102, 102, 156]
    # colormap[4] = [190, 153, 153]
    # colormap[5] = [153, 153, 153]
    # colormap[6] = [250, 170, 30]
    # colormap[7] = [220, 220, 0]
    # colormap[8] = [107, 142, 35]
    # colormap[9] = [152, 251, 152]
    # colormap[10] = [70, 130, 180]
    # colormap[11] = [220, 20, 60]
    # colormap[12] = [255, 0, 0]
    # colormap[13] = [0, 0, 142]
    # colormap[14] = [0, 0, 70]
    # colormap[15] = [0, 60, 100]
    # colormap[16] = [0, 80, 100]
    # colormap[17] = [0, 0, 230]
    # colormap[18] = [119, 11, 32]
    # return colormap

    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 255, 0]  ## road  green
    colormap[1] = [0, 255, 0]  ## road  green
    colormap[2] = [0, 0, 0]  ## background black
    colormap[3] = [0, 0, 0]  ## background black
    colormap[4] = [0, 0, 0]  ## background black
    colormap[5] = [0, 0, 0]  ## background black
    colormap[6] = [0, 0, 0]  ## background black
    colormap[7] = [0, 0, 0]  ## background black
    colormap[8] = [0, 0, 0]  ## background black
    colormap[9] = [0, 0, 0]  ## background black
    colormap[10] = [0, 0, 0]  ## background black
    colormap[11] = [0, 0, 255]  ## person
    colormap[12] = [0, 0, 0]  ## background black
    colormap[13] = [255, 0, 0]  ## vehicle  red
    colormap[14] = [255, 0, 0]  ## vehicle  red
    colormap[15] = [255, 0, 0]  ## vehicle  red
    colormap[16] = [255, 0, 0]  ## vehicle  red
    colormap[17] = [255, 0, 0]  ## vehicle  red
    colormap[18] = [255, 0, 0]  ## vehicle  red

    ## color for pedestrian : blue

    return colormap


def label_to_color_image(label):
    colormap = create_cityscapes_label_colormap()
    return colormap[label]


def init_canvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    (channel_b, channel_g, channel_r) = cv2.split(canvas)
    channel_b *= color[0]
    channel_g *= color[1]
    channel_r *= color[2]
    return cv2.merge([channel_b, channel_g, channel_r])


# def vis_segmentation(image, seg_map):
#     """Visualizes input image, segmentation map and overlay view."""
#     plt.figure(figsize=(15, 5))
#     grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
#
#     plt.subplot(grid_spec[0])
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title('input image')
#
#     plt.subplot(grid_spec[1])
#     seg_image = label_to_color_image(seg_map).astype(np.uint8)
#     plt.imshow(seg_image)
#     plt.axis('off')
#     plt.title('segmentation map')
#
#     plt.subplot(grid_spec[2])
#     plt.imshow(image)
#     plt.imshow(seg_image, alpha=0.7)
#     plt.axis('off')
#     plt.title('segmentation overlay')
#
#     LABEL_NAMES = np.asarray([
#         'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
#         'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
#         'bus', 'train', 'motorcycle', 'bycycle'])
#
#     FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
#     FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
#
#     unique_labels = np.unique(seg_map)
#     ax = plt.subplot(grid_spec[3])
#     plt.imshow(
#         FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
#     ax.yaxis.tick_right()
#     plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
#     plt.xticks([], [])
#     ax.tick_params(width=0.0)
#     plt.grid('off')
#     plt.show()


################################################


# TODO: Make all global class parameters to minimum , e.g. no model generation
class ActionPredict:
    """
        A base interface class for creating prediction models
    """

    def __init__(self,
                 global_pooling='avg',
                 regularizer_val=0.0001,
                 backbone='vgg19',
                 **kwargs):
        """
        Class init function
        Args:
            global_pooling: Pooling method for generating convolutional features
            regularizer_val: Regularization value for training
            backbone: Backbone for generating convolutional features
        """
        # Network parameters
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)
        self._global_pooling = global_pooling
        self._backbone = backbone
        self._generator = None  # use data generator for train/test

    # Processing images anf generate features
    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     process=True,
                                     regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_mode: How to resize ond/or pad the cropped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            process: Whether process the raw images using a neural network
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """
        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
              \nsave_path={}, backbone={}, process={} ".format(data_type, crop_type, crop_mode,
                                                               save_path, self._backbone, str(process)))

        # backbone to savepath
        save_path = os.path.join(save_path, self._backbone)

        # load segmentation model
        segmodel_path = "deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz"
        segmodel = DeepLabModel(segmodel_path)
        LABEL_NAMES = np.asarray([
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
            'bus', 'train', 'motorcycle', 'bycycle'])

        FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
        FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
        ##########################
        preprocess_dict = {'vgg19': vgg19.preprocess_input, 'resnet152': resnet.preprocess_input,
                           'efficientnet': efficientnet.preprocess_input, 'mobilenet_v2': mobilenet_v2.preprocess_input}

        backbone_dict = {'vgg19': vgg19.VGG19, 'resnet152': resnet.ResNet152,
                         'efficientnet': efficientnet.EfficientNetB7, 'mobilenet_v2': mobilenet_v2.MobileNetV2}

        model_inputs = {'input_shape': (224, 224, 3), 'weights': 'imagenet', 'include_top': False}
        vit_model_inputs = {'image_size': 224, 'pretrained': True, 'include_top': False, 'pretrained_top': False}

        base_model = backbone_dict[self._backbone](**model_inputs)
        vit_model = vit.vit_b16(**vit_model_inputs)

        backbone_model = base_model
        # backbone_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

        preprocess_input = preprocess_dict.get(self._backbone, None)

        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id, vid_id, img_name = PurePath(imp).parts[-3:]
                img_name = img_name.split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                # Modify the path depending on crop mode
                if crop_type == 'none':
                    img_save_path = os.path.join(img_save_folder, img_name + '_n1.pkl')
                else:
                    img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                # Check whether the file exists
                if os.path.exists(img_save_path) and not regen_data:
                    if not self._generator:
                        with open(img_save_path, 'rb') as fid:
                            try:
                                img_features = pickle.load(fid)
                            except:
                                img_features = pickle.load(fid, encoding='bytes')
                else:
                    if 'flip' in imp:
                        imp = imp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        img_data = cv2.imread(imp)
                        img_features = cv2.resize(img_data, target_dim)
                        if flip_image:
                            img_features = cv2.flip(img_features, 1)
                    elif crop_type == 'local_context_cnn':
                        img = image.load_img(imp, target_size=(224, 224))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)
                        img_features = backbone_model.predict(x)
                        img_features = tf.nn.avg_pool2d(img_features, ksize=[7, 7], strides=[1, 1, 1, 1], # ksize=[14, 14]
                                                        padding='VALID') # check output size
                        img_features = tf.squeeze(img_features)
                        # with tf.compact.v1.Session():
                        img_features = img_features.numpy()

                    elif crop_type == 'mask_cnn':
                        img_data = cv2.imread(imp)
                        ori_dim = img_data.shape
                        # bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                        # bbox = squarify(bbox, 1, img_data.shape[1])
                        # bbox = list(map(int, bbox[0:4]))
                        b = list(map(int, b[0:4]))
                        ## img_data --- > mask_img_data (deeplabV3)
                        original_im = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                        resized_im, seg_map = segmodel.run(original_im)
                        resized_im = np.array(resized_im)
                        seg_image = label_to_color_image(seg_map).astype(np.uint8)
                        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
                        seg_image = cv2.addWeighted(resized_im, 0.5, seg_image, 0.5, 0)
                        img_data = cv2.resize(seg_image, (ori_dim[1], ori_dim[0]))
                        ## mask_img_data + pd highlight ---> final_mask_img_data
                        ped_mask = init_canvas(b[2] - b[0], b[3] - b[1], color=(255, 255, 255))
                        img_data[b[1]:b[3], b[0]:b[2]] = ped_mask
                        img_features = cv2.resize(img_data, target_dim)
                        img = Image.fromarray(cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)
                        img_features = backbone_model.predict(x)
                        img_features = tf.nn.avg_pool2d(img_features, ksize=[7, 7], strides=[1, 1, 1, 1], #todo adapt for 14,14? ksize=[14, 14]
                                                        padding='VALID')
                        img_features = tf.squeeze(img_features)
                        # with tf.compact.v1.Session():
                        img_features = img_features.numpy()
                        if flip_image:
                            img_features = cv2.flip(img_features, 1)

                    elif crop_type == 'mask_vit':
                        img_data = cv2.imread(imp)
                        ori_dim = img_data.shape
                        # bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                        # bbox = squarify(bbox, 1, img_data.shape[1])
                        # bbox = list(map(int, bbox[0:4]))
                        b = list(map(int, b[0:4]))
                        ## img_data --- > mask_img_data (deeplabV3)
                        original_im = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                        resized_im, seg_map = segmodel.run(original_im)
                        resized_im = np.array(resized_im)
                        seg_image = label_to_color_image(seg_map).astype(np.uint8)
                        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
                        seg_image = cv2.addWeighted(resized_im, 0.5, seg_image, 0.5, 0)
                        img_data = cv2.resize(seg_image, (ori_dim[1], ori_dim[0]))
                        ## mask_img_data + pd highlight ---> final_mask_img_data
                        ped_mask = init_canvas(b[2] - b[0], b[3] - b[1], color=(255, 255, 255))
                        img_data[b[1]:b[3], b[0]:b[2]] = ped_mask
                        img_features = cv2.resize(img_data, target_dim)
                        img = Image.fromarray(cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = vit.preprocess_inputs(x) # ! global uses vit / local still uses default cnn
                        img_features = vit_model.predict(x) # ! generalize somehow? line above as well TODO fix inputs of next layer (768 out)
                        img_features = tf.squeeze(img_features)
                        img_features = img_features.numpy()
                        if flip_image:
                            img_features = cv2.flip(img_features, 1)

                    elif crop_type == 'context_split':
                        img_data = cv2.imread(imp)
                        ori_dim = img_data.shape
                        b = list(map(int, b[0:4]))
                        ## img_data --- > mask_img_data (deeplabV3)
                        original_im = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                        resized_im, seg_map = segmodel.run(original_im)
                        resized_im = np.array(resized_im)
                        seg_image = label_to_color_image(seg_map).astype(np.uint8)
                        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
                        seg_image = cv2.addWeighted(resized_im, 0.5, seg_image, 0.5, 0)
                        img_data = cv2.resize(seg_image, (ori_dim[1], ori_dim[0]))
                        ## mask_img_data + pd highlight ---> final_mask_img_data
                        ped_mask = init_canvas(b[2] - b[0], b[3] - b[1], color=(255, 255, 255))
                        img_data[b[1]:b[3], b[0]:b[2]] = ped_mask
                        img_features = cv2.resize(img_data, (target_dim[0]*2, target_dim[1]*2))
                        img = Image.fromarray(cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB))
                        img = image.img_to_array(img)
                        patches_coords = [(0, 1, 0, 1), (1, 2, 0, 1), (0, 1, 1, 2), (1, 2, 1, 2)]  # split scene into 4 patches
                        hdim0 = img.shape[0]//2
                        hdim1 = img.shape[1]//2
                        patches = (img[hdim0*xa:hdim0*xb, hdim1*ya:hdim1*yb] for xa, xb, ya, yb in patches_coords)
                        img_features = []
                        for x in patches:
                            x = np.expand_dims(x, axis=0)
                            x = preprocess_input(x)
                            feat = backbone_model.predict(x)
                            feat = tf.nn.avg_pool2d(feat, ksize=[7, 7], strides=[1, 1, 1, 1],
                                                        padding='VALID')
                            feat = tf.squeeze(feat)
                            # with tf.compact.v1.Session():
                            feat = feat.numpy()
                            if flip_image:
                                feat = cv2.flip(feat, 1)
                            img_features.append(feat)
                        img_features = np.array(img_features)

                    else:
                        img_data = cv2.imread(imp)
                        if flip_image:
                            img_data = cv2.flip(img_data, 1)
                        if crop_type == 'bbox':
                            b = list(map(int, b[0:4]))
                            cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]
                            img_features = img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        elif 'surround' in crop_type:
                            b_org = list(map(int, b[0:4])).copy()
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            img_data[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                    # Save the file
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

                if self._generator:
                    img_seq.append(img_save_path)
                else:
                    img_seq.append(img_features)
            sequences.append(img_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            with open(sequences[0][0], 'rb') as fid:
                feat_shape = pickle.load(fid).shape
            if process:
                if self._global_pooling in ['max', 'avg']:
                    feat_shape = feat_shape[-1]
                else:
                    feat_shape = np.prod(feat_shape)
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]

        return sequences, feat_shape

        # Processing images anf generate features

    def get_optical_flow(self, img_sequences, bbox_sequences,
                         ped_ids, save_path,
                         data_type='train',
                         crop_type='none',
                         crop_mode='warp',
                         crop_resize_ratio=2,
                         target_dim=(224, 224),
                         regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_mode: How to resize ond/or pad the corpped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
               \nsave_path={}, ".format(data_type, crop_type, crop_mode, save_path))
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        # flow size (h,w)
        flow_size = read_flow_file(img_sequences[0][0].replace('images', 'optical_flow').replace('png', 'flo')).shape
        img_size = cv2.imread(img_sequences[0][0]).shape
        # A ratio to adjust the dimension of bounding boxes (w,h)
        box_resize_coef = (flow_size[1] / img_size[1], flow_size[0] / img_size[0])

        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            flow_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id, vid_id, img_name = PurePath(imp)[-3:]
                set_id = set_id.split('.')[0]
                optflow_save_folder = os.path.join(save_path, set_id, vid_id)
                ofp = imp.replace('images', 'optical_flow').replace('png', 'flo')
                # Modify the path depending on crop mode
                if crop_type == 'none':
                    optflow_save_path = os.path.join(optflow_save_folder, img_name + '.flo')
                else:
                    optflow_save_path = os.path.join(optflow_save_folder, img_name + '_' + p[0] + '.flo')

                # Check whether the file exists
                if os.path.exists(optflow_save_path) and not regen_data:
                    if not self._generator:
                        ofp_data = read_flow_file(optflow_save_path)
                else:
                    if 'flip' in imp:
                        ofp = ofp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        ofp_image = read_flow_file(ofp)
                        ofp_data = cv2.resize(ofp_image, target_dim)
                        if flip_image:
                            ofp_data = cv2.flip(ofp_data, 1)
                    else:
                        ofp_image = read_flow_file(ofp)
                        # Adjust the size of bbox according to the dimensions of flow map
                        b = list(map(int, [b[0] * box_resize_coef[0], b[1] * box_resize_coef[1],
                                           b[2] * box_resize_coef[0], b[3] * box_resize_coef[1]]))
                        if flip_image:
                            ofp_image = cv2.flip(ofp_image, 1)
                        if crop_type == 'bbox':
                            cropped_image = ofp_image[b[1]:b[3], b[0]:b[2], :]
                            ofp_data = img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, ofp_image.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = ofp_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            ofp_data = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        elif 'surround' in crop_type:
                            b_org = b.copy()
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, ofp_image.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            ofp_image[b_org[1]:b_org[3], b_org[0]: b_org[2], :] = 0
                            cropped_image = ofp_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            ofp_data = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))

                    # Save the file
                    if not os.path.exists(optflow_save_folder):
                        os.makedirs(optflow_save_folder)
                    write_flow(ofp_data, optflow_save_path)

                # if using the generator save the cached features path and size of the features
                if self._generator:
                    flow_seq.append(optflow_save_path)
                else:
                    flow_seq.append(ofp_data)
            sequences.append(flow_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            feat_shape = read_flow_file(sequences[0][0]).shape
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]
        return sequences, feat_shape

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        Generates raw sequences from a given dataset
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            opts:  Options for generating data samples
        Returns:
            A list of data samples extracted from raw data
            Positive and negative data counts
        """
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]] * len(data_raw['bbox'])
        else:
            overlap = opts['overlap']  # if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs

            for seq in data_raw['bbox']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
        if normalize:
            for k in d.keys():
                if k != 'tte':
                    if k != 'box' and k != 'center':
                        for i in range(len(d[k])):
                            d[k][i] = d[k][i][1:]
                    else:
                        for i in range(len(d[k])):
                            d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                d[k] = np.array(d[k])
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def get_context_data(self, model_opts, data, data_type, feature_type):
        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')
        process = model_opts.get('process', True)
        aux_name = [self._backbone]
        if not process:
            aux_name.append('raw')
        aux_name = '_'.join(aux_name).strip('_')
        eratio = model_opts['enlarge_ratio']
        dataset = model_opts['dataset']

        data_gen_params = {'data_type': data_type, 'crop_type': 'none',
                           'target_dim': model_opts.get('target_dim', (224, 224))}
        if 'local_box' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif 'mask_cnn' in feature_type:
            data_gen_params['crop_type'] = 'mask_cnn'
        elif 'mask_vit' in feature_type:
            data_gen_params['crop_type'] = 'mask_vit'
        elif 'mask' in feature_type:
            data_gen_params['crop_type'] = 'mask'
            # data_gen_params['crop_mode'] = 'pad_resize'
        elif 'local_context_cnn' in feature_type:
            data_gen_params['crop_type'] = 'local_context_cnn'
        elif 'local_context' in feature_type:
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'surround' in feature_type:
            data_gen_params['crop_type'] = 'surround'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'scene_context' in feature_type:
            data_gen_params['crop_type'] = 'none'
        save_folder_name = feature_type
        if 'flow' not in feature_type:
            save_folder_name = '_'.join([feature_type, aux_name])
            if 'local_context' in feature_type or 'surround' in feature_type:
                save_folder_name = '_'.join([save_folder_name, str(eratio)])
        data_gen_params['save_path'], _ = get_path(save_folder=save_folder_name,
                                                   dataset=dataset, save_root_folder='data/features')
        if 'flow' in feature_type:
            return self.get_optical_flow(data['image'],
                                         data['box_org'],
                                         data['ped_id'],
                                         **data_gen_params)
        else:
            return self.load_images_crop_and_process(data['image'],
                                                     data['box_org'],
                                                     data['ped_id'],
                                                     process=process,
                                                     **data_gen_params)

    def get_data(self, data_type, data_raw, model_opts):
        """
        Generates data train/test/val data
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            model_opts: Model options for generating data
        Returns:
            A dictionary containing, data, data parameters used for model generation,
            effective dimension of data (the number of rgb images to be used calculated accorfing
            to the length of optical flow window) and negative and positive sample counts
        """

        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type or 'mask' in d_type:
                features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def log_configs(self, config_path, batch_size, epochs,
                    lr, model_opts):

        # TODO: Update config by adding network attributes
        """
        Logs the parameters of the model and training
        Args:
            config_path: The path to save the file
            batch_size: Batch size of training
            epochs: Number of epochs for training
            lr: Learning rate of training
            model_opts: Data generation parameters (see get_data)
        """
        # Save config and training param files
        with open(config_path, 'wt') as fid:
            yaml.dump({'model_opts': model_opts,
                       'train_opts': {'batch_size': batch_size, 'epochs': epochs, 'lr': lr}},
                      fid, default_flow_style=False)
        # with open(config_path, 'wt') as fid:
        #     fid.write("####### Model options #######\n")
        #     for k in opts:
        #         fid.write("%s: %s\n" % (k, str(opts[k])))

        #     fid.write("\n####### Network config #######\n")
        #     # fid.write("%s: %s\n" % ('hidden_units', str(self._num_hidden_units)))
        #     # fid.write("%s: %s\n" % ('reg_value ', str(self._regularizer_value)))

        #     fid.write("\n####### Training config #######\n")
        #     fid.write("%s: %s\n" % ('batch_size', str(batch_size)))
        #     fid.write("%s: %s\n" % ('epochs', str(epochs)))
        #     fid.write("%s: %s\n" % ('lr', str(lr)))

        print('Wrote configs to {}'.format(config_path))

    def class_weights(self, apply_weights, sample_count):
        """
        Computes class weights for imbalanced data used during training
        Args:
            apply_weights: Whether to apply weights
            sample_count: Positive and negative sample counts
        Returns:
            A dictionary of class weights or None if no weights to be calculated
        """
        if not apply_weights:
            return None

        total = sample_count['neg_count'] + sample_count['pos_count']
        # formula from sklearn
        # neg_weight = (1 / sample_count['neg_count']) * (total) / 2.0
        # pos_weight = (1 / sample_count['pos_count']) * (total) / 2.0

        # use simple ratio
        neg_weight = sample_count['pos_count'] / total
        pos_weight = sample_count['neg_count'] / total

        print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
        return {0: neg_weight, 1: pos_weight}

    def get_callbacks(self, learning_scheduler, model_path):
        """
        Creates a list of callabcks for training
        Args:
            learning_scheduler: Whether to use callbacks
        Returns:
            A list of call backs or None if learning_scheduler is false
        """
        wandb_callback = WandbCallback(log_evaluation=True)
        callbacks = [wandb_callback]

        # Set up learning schedulers
        if learning_scheduler:
            if 'early_stop' in learning_scheduler:
                default_params = {'monitor': 'val_loss',
                                  'min_delta': 1, 'patience': 5,
                                  'verbose': 1}
                default_params.update(learning_scheduler['early_stop'])
                print(f'using early stopping: {default_params}')
                callbacks.append(EarlyStopping(**default_params))

            if 'plateau' in learning_scheduler:
                default_params = {'monitor': 'val_loss',
                                  'factor': 0.2, 'patience': 5,
                                  'min_lr': 1e-08, 'verbose': 1}
                default_params.update(learning_scheduler['plateau'])
                callbacks.append(ReduceLROnPlateau(**default_params))

            if 'checkpoint' in learning_scheduler:
                default_params = {'filepath': model_path, 'monitor': 'val_loss',
                                  'save_best_only': True, 'save_weights_only': False,
                                  'save_freq': 'epoch', 'verbose': 2}
                default_params.update(learning_scheduler['checkpoint'])
                callbacks.append(ModelCheckpoint(**default_params))

        return callbacks

    def get_optimizer(self, optimizer):
        """
        Return an optimizer object
        Args:
            optimizer: The type of optimizer. Supports 'adam', 'sgd', 'rmsprop'
        Returns:
            An optimizer object
        """
        assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
            "{} optimizer is not implemented".format(optimizer)
        if optimizer.lower() == 'adam':
            return Adam
        elif optimizer.lower() == 'sgd':
            return SGD
        elif optimizer.lower() == 'rmsprop':
            return RMSprop

    def train(self, data_train,
              data_val,
              batch_size=2,
              epochs=60,
              lr=0.000005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]

        # Create model
        train_model = self.get_model(data_train['data_params'])

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(learning_rate=lr)
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        ## reivse fit

        callbacks = self.get_callbacks(learning_scheduler, model_path)

        # data_val = data_val.batch(batch_size)
        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=None,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    # Test Functions
    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)
            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')

        test_model = load_model(os.path.join(model_path, 'model.h5'))
        test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})

        test_results = test_model.predict(test_data['data'][0],
                                          batch_size=1, verbose=1)
        acc = accuracy_score(test_data['data'][1], np.round(test_results))
        f1 = f1_score(test_data['data'][1], np.round(test_results))
        auc = roc_auc_score(test_data['data'][1], np.round(test_results))
        roc = roc_curve(test_data['data'][1], test_results)
        precision = precision_score(test_data['data'][1], np.round(test_results))
        recall = recall_score(test_data['data'][1], np.round(test_results))
        pre_recall = precision_recall_curve(test_data['data'][1], test_results)

        # todo THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                         'pid': test_data['ped_id'],
                         'gt': test_data['data'][1],
                         'y': test_results,
                         'image': test_data['image']}, picklefile)

        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)
        return acc, auc, f1, precision, recall

    def get_model(self, data_params):
        """
        Generates a model
        Args:
            data_params: Data parameters to use for model generation
        Returns:
            A model
        """
        raise NotImplementedError("get_model should be implemented")

    # Auxiliary function
    def _gru(self, name='gru', r_state=False, r_sequence=False):
        """
        A helper function to create a single GRU unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the GRU
            r_sequence: Whether to return a sequence
        Return:
            A GRU unit
        """
        return GRU(units=self._num_hidden_units,
                   return_state=r_state,
                   return_sequences=r_sequence,
                   stateful=False,
                   kernel_regularizer=self._regularizer,
                   recurrent_regularizer=self._regularizer,
                   bias_regularizer=self._regularizer,
                   name=name)

    def _lstm(self, name='lstm', r_state=False, r_sequence=False):
        """
        A helper function to create a single LSTM unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the LSTM
            r_sequence: Whether to return a sequence
        Return:
            A LSTM unit
        """
        return LSTM(units=self._num_hidden_units,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    stateful=False,
                    kernel_regularizer=self._regularizer,
                    recurrent_regularizer=self._regularizer,
                    bias_regularizer=self._regularizer,
                    name=name)

    def create_stack_rnn(self, size, r_state=False, r_sequence=False):
        """
        Creates a stack of recurrent cells
        Args:
            size: The size of stack
            r_state: Whether to return the states of the GRU
            r_sequence: Whether the last stack layer to return a sequence
        Returns:
            A stacked recurrent model
        """
        cells = []
        for i in range(size):
            cells.append(self._rnn_cell(units=self._num_hidden_units,
                                        kernel_regularizer=self._regularizer,
                                        recurrent_regularizer=self._regularizer,
                                        bias_regularizer=self._regularizer, ))
        return RNN(cells, return_sequences=r_sequence, return_state=r_state)


def attention_3d_block(hidden_states, dense_size=128, modality=''):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    @author: felixhao28.
    """
    print(f'hidden_states:{hidden_states.shape}')
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec' + modality)(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state' + modality)(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score' + modality)
    attention_weights = Activation('softmax', name='attention_weight' + modality)(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector' + modality)
    pre_activation = concatenate([context_vector, h_t], name='attention_output' + modality)
    attention_vector = Dense(dense_size, use_bias=False, activation='tanh', name='attention_vector' + modality)(
        pre_activation)
    return attention_vector


class MASK_PCPA_4_2D(ActionPredict):
    """
    hierfusion MASK_PCPA
    Class init function

    Args:
        num_hidden_units: Number of recurrent hidden layers
        cell_type: Type of RNN cell
        **kwargs: Description
    """

    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]
        # if 'context_cnn' in data.keys():
        #     data_type_sizes_dict['context_cnn'] = data['context_cnn'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type or 'mask' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=None,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        # global_pooling=self._global_pooling,
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        # conv3d_model = self._3dconv()
        # network_inputs.append(conv3d_model.input)
        #
        attention_size = self._num_hidden_units
        for i in range(0, core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

        x = self._rnn(name='enc0_' + data_types[0], r_sequence=return_sequence)(network_inputs[0])
        encoder_outputs.append(x)
        x = self._rnn(name='enc1_' + data_types[1], r_sequence=return_sequence)(network_inputs[1])
        encoder_outputs.append(x)
        x = self._rnn(name='enc2_' + data_types[2], r_sequence=return_sequence)(network_inputs[2])
        current = [x, network_inputs[3]]
        x = Concatenate(name='concat_early3', axis=2)(current)
        x = self._rnn(name='enc3_' + data_types[3], r_sequence=return_sequence)(x)
        current = [x, network_inputs[4]]
        x = Concatenate(name='concat_early4', axis=2)(current)
        x = self._rnn(name='enc4_' + data_types[4], r_sequence=return_sequence)(x)
        encoder_outputs.append(x)

        if len(encoder_outputs) > 1:
            att_enc_out = []
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[0:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_' + data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

        else:
            encodings = encoder_outputs[0]
            encodings = attention_3d_block(encodings, dense_size=attention_size, modality='_modality')

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        net_model.summary()
        plot_model(net_model, to_file='MASK_PCPA_4_2D.png')
        return net_model


class MASK_PCPA_4_2D_NOGLOBAL(ActionPredict):
    """
    hierfusion MASK_PCPA
    Class init function

    Args:
        num_hidden_units: Number of recurrent hidden layers
        cell_type: Type of RNN cell
        **kwargs: Description
    """

    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell
        # assert self._backbone in ['c3d', 'i3d'], 'Incorrect backbone {}! Should be C3D or I3D'.format(self._backbone)
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet

        # dropout = 0.0,
        # dense_activation = 'sigmoid',
        # freeze_conv_layers = False,
        # weights = 'imagenet',
        # num_classes = 1,
        # backbone = 'vgg16',

        # self._dropout = dropout
        # self._dense_activation = dense_activation
        # self._freeze_conv_layers = False
        # self._weights = 'imagenet'
        # self._num_classes = 1
        # self._conv_models = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50, 'alexnet': AlexNet}
        # self._backbone ='vgg16'

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]
        # if 'context_cnn' in data.keys():
        #     data_type_sizes_dict['context_cnn'] = data['context_cnn'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type or 'mask' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=None,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        # global_pooling=self._global_pooling,
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        # conv3d_model = self._3dconv()
        # network_inputs.append(conv3d_model.input)
        #
        attention_size = self._num_hidden_units
        for i in range(0, core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

        # x = self._rnn(name='enc0_' + data_types[0], r_sequence=return_sequence)(network_inputs[0])  # global
        # encoder_outputs.append(x)
        x = self._rnn(name='enc1_' + data_types[0], r_sequence=return_sequence)(network_inputs[0])  # local
        encoder_outputs.append(x)
        x = self._rnn(name='enc2_' + data_types[1], r_sequence=return_sequence)(network_inputs[1])  # pose
        current = [x, network_inputs[2]]  # concat pose_rnn + bounding boxes
        x = Concatenate(name='concat_early3', axis=2)(current)
        x = self._rnn(name='enc3_' + data_types[2], r_sequence=return_sequence)(x)
        current = [x, network_inputs[3]]  # concat pose-bb_rnn + vehicle_speed
        x = Concatenate(name='concat_early4', axis=2)(current)
        x = self._rnn(name='enc4_' + data_types[3], r_sequence=return_sequence)(x)
        encoder_outputs.append(x)

        if len(encoder_outputs) > 1:
            att_enc_out = []
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[0:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_' + data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

            # print(encodings.shape)
            # print(weights_softmax.shape)
        else:
            encodings = encoder_outputs[0]
            encodings = attention_3d_block(encodings, dense_size=attention_size, modality='_modality')

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        net_model.summary()
        plot_model(net_model, to_file='MASK_PCPA_4_2D_NOGLOBAL.png')
        return net_model

class MASK_PCPA_4_2D_CNN(ActionPredict):
    """
    hierfusion MASK_PCPA
    Class init function

    Args:
        num_hidden_units: Number of recurrent hidden layers
        cell_type: Type of RNN cell
        **kwargs: Description
    """

    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell
        self._3dconv = SIMPLE_CNN

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]
        # if 'context_cnn' in data.keys():
        #     data_type_sizes_dict['context_cnn'] = data['context_cnn'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type or 'mask' in d_type or 'scene' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=None,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        # global_pooling=self._global_pooling,
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        # conv3d_model = self._3dconv()
        # network_inputs.append(conv3d_model.input)
        #
        attention_size = self._num_hidden_units
        for i in range(0, core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

        x = self._rnn(name='enc0_' + data_types[0], r_sequence=return_sequence)(network_inputs[0])
        encoder_outputs.append(x)
        x = self._3dconv()(network_inputs[1])
        x = self._rnn(name='enc1_' + data_types[1], r_sequence=return_sequence)(tf.squeeze(x, [2,3]))
        encoder_outputs.append(x)
        x = self._rnn(name='enc2_' + data_types[2], r_sequence=return_sequence)(network_inputs[2])
        current = [x, network_inputs[3]]
        x = Concatenate(name='concat_early3', axis=2)(current)
        x = self._rnn(name='enc3_' + data_types[3], r_sequence=return_sequence)(x)
        current = [x, network_inputs[4]]
        x = Concatenate(name='concat_early4', axis=2)(current)
        x = self._rnn(name='enc4_' + data_types[4], r_sequence=return_sequence)(x)
        encoder_outputs.append(x)

        if len(encoder_outputs) > 1:
            att_enc_out = []
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[0:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_' + data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

        else:
            encodings = encoder_outputs[0]
            encodings = attention_3d_block(encodings, dense_size=attention_size, modality='_modality')

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        net_model.summary()
        plot_model(net_model, to_file='MASK_PCPA_4_2D_CNN.png')
        return net_model


class MASK_PCPA_4_2D_SPLIT(ActionPredict):
    """
    hierfusion MASK_PCPA_SPLIT
    Class init function

    Args:
        num_hidden_units: Number of recurrent hidden layers
        cell_type: Type of RNN cell
        **kwargs: Description
    """

    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]
        # if 'context_cnn' in data.keys():
        #     data_type_sizes_dict['context_cnn'] = data['context_cnn'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type or 'mask' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=None,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        # global_pooling=self._global_pooling,
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        # conv3d_model = self._3dconv()
        # network_inputs.append(conv3d_model.input)
        #
        attention_size = self._num_hidden_units
        print(data_sizes[1])
        for i in range(0, core_size):
            if i == 1:
                network_inputs.append(Input(shape=data_sizes[i][0], name='input_' + data_types[i]))
            else:
                network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

        x = self._rnn(name='enc0_' + data_types[0], r_sequence=return_sequence)(network_inputs[0])
        encoder_outputs.append(x)

        att_inputs = []
        # for recurrent branches apply many-to-one attention block
        x = attention_3d_block(network_inputs[1][0], dense_size=attention_size, modality='_' + data_types[1]) #lel
        x = Dropout(0.5)(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
        att_inputs.append(x)
        x = attention_3d_block(network_inputs[1][1], dense_size=attention_size, modality='_' + data_types[1])
        x = Dropout(0.5)(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
        att_inputs.append(x)
        x = attention_3d_block(network_inputs[1][2], dense_size=attention_size, modality='_' + data_types[1])
        x = Dropout(0.5)(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
        att_inputs.append(x)
        x = attention_3d_block(network_inputs[1][3], dense_size=attention_size, modality='_' + data_types[1])
        x = Dropout(0.5)(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
        att_inputs.append(x)

        x = Concatenate(name='concat_modalities', axis=1)(att_inputs)
        x = self._rnn(name='enc1_' + data_types[1], r_sequence=return_sequence)(x)
        encoder_outputs.append(x)

        x = self._rnn(name='enc2_' + data_types[2], r_sequence=return_sequence)(network_inputs[2])
        current = [x, network_inputs[3]]
        x = Concatenate(name='concat_early3', axis=2)(current)
        x = self._rnn(name='enc3_' + data_types[3], r_sequence=return_sequence)(x)
        current = [x, network_inputs[4]]
        x = Concatenate(name='concat_early4', axis=2)(current)
        x = self._rnn(name='enc4_' + data_types[4], r_sequence=return_sequence)(x)
        encoder_outputs.append(x)

        if len(encoder_outputs) > 1:
            att_enc_out = []
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[0:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_' + data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

        else:
            encodings = encoder_outputs[0]
            encodings = attention_3d_block(encodings, dense_size=attention_size, modality='_modality')

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        net_model.summary()
        plot_model(net_model, to_file='MASK_PCPA_4_2D_SPLIT.png')
        return net_model


def action_prediction(model_name):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))


class MASK_PCPA_4_2D_ViT(ActionPredict):
    """
    HYBRID MASK_PCPA_ViT

    Args:
        num_hidden_units: Number of recurrent hidden layers
        cell_type: Type of RNN cell
        **kwargs: Description
    """

    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function

        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]
        # if 'context_cnn' in data.keys():
        #     data_type_sizes_dict['context_cnn'] = data['context_cnn'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type or 'mask' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=None,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        # global_pooling=self._global_pooling,
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        # conv3d_model = self._3dconv()
        # network_inputs.append(conv3d_model.input)
        #
        attention_size = self._num_hidden_units
        for i in range(0, core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

        x = self._rnn(name='enc0_' + data_types[0], r_sequence=return_sequence)(network_inputs[0])
        encoder_outputs.append(x)
        x = self._rnn(name='enc1_' + data_types[1], r_sequence=return_sequence)(network_inputs[1])
        encoder_outputs.append(x)
        x = self._rnn(name='enc2_' + data_types[2], r_sequence=return_sequence)(network_inputs[2])
        current = [x, network_inputs[3]]
        x = Concatenate(name='concat_early3', axis=2)(current)
        x = self._rnn(name='enc3_' + data_types[3], r_sequence=return_sequence)(x)
        current = [x, network_inputs[4]]
        x = Concatenate(name='concat_early4', axis=2)(current)
        x = self._rnn(name='enc4_' + data_types[4], r_sequence=return_sequence)(x)
        encoder_outputs.append(x)

        if len(encoder_outputs) > 1:
            att_enc_out = []
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[0:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_' + data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

        else:
            encodings = encoder_outputs[0]
            encodings = attention_3d_block(encodings, dense_size=attention_size, modality='_modality')

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        net_model.summary()
        plot_model(net_model, to_file='MASK_PCPA_4_2D_ViT.png')
        return net_model


class DataGenerator(Sequence):

    def __init__(self,
                 data=None,
                 labels=None,
                 data_sizes=None,
                 process=False,
                 global_pooling=None,
                 input_type_list=None,
                 batch_size=32,
                 shuffle=True,
                 to_fit=True,
                 stack_feats=False):
        self.data = data
        self.labels = labels
        self.process = process
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list
        self.batch_size = 1 if len(self.labels) < batch_size else batch_size
        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data[0]) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        X = self._generate_X(indices)
        if self.to_fit:
            y = self._generate_y(indices)
            return X, y
        else:
            return X

    def _get_img_features(self, cached_path):
        with open(cached_path, 'rb') as fid:
            try:
                img_features = pickle.load(fid)
            except:
                img_features = pickle.load(fid, encoding='bytes')
        if self.process:
            if self.global_pooling == 'max':
                img_features = np.squeeze(img_features)
                img_features = np.amax(img_features, axis=0)
                img_features = np.amax(img_features, axis=0)
            elif self.global_pooling == 'avg':
                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
            else:
                img_features = img_features.ravel()
        return img_features

    def _generate_X(self, indices):
        X = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, *self.data_sizes[input_type_idx]))
            num_ch = features_batch.shape[-1] // len(self.data[input_type_idx][0])
            for i, index in enumerate(indices):
                if isinstance(self.data[input_type_idx][index][0], str):
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        if 'flow' in input_type:
                            img_features = read_flow_file(cached_path)
                        else:
                            img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            # for static model if only one image in the sequence
                            features_batch[i,] = img_features
                        else:
                            if self.stack_feats and 'flow' in input_type:
                                features_batch[i, ..., j * num_ch:j * num_ch + num_ch] = img_features
                            elif 'scene_context' in input_type:
                                features_batch[i, j,] = tf.reshape(img_features, (224,224,3))  # todo fix this custom size
                            else:
                                features_batch[i, j,] = img_features
                else:
                    features_batch[i,] = self.data[input_type_idx][index]
            X.append(features_batch)
        return X

    def _generate_y(self, indices):
        return np.array(self.labels[indices])
