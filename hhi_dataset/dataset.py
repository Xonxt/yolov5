##
# ingroup: DatasetAnnotation
# file:    dataset.py
# brief:   This is a wrapper class that reads and interacts with the HHI Json Dataset Format
# author:  Nikita Kovalenko (mykyta.kovalenko@hhi.fraunhofer.de)
# date:    01.10.2020
#
# Copyright:
# 2020 Fraunhofer Institute for Telecommunications, Heinrich-Hertz-Institut (HHI)
# The copyright of this software source code is the property of HHI.
# This software may be used and/or copied only with the written permission
# of HHI and in accordance with the terms and conditions stipulated
# in the agreement/contract under which the software has been supplied.
# The software distributed under this license is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either expressed or implied.
##

## --------------------------------------------------------------------------------------------------------------------------
import os, sys
import numpy as np
import cv2
import random
import math
import json
import time
import shutil
import copy
# -------------------------------------
from operator import itemgetter
# -------------------------------------
from hhi_dataset.tools import *

#----------------------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types
    Source: https://stackoverflow.com/a/49677241
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
#----------------------------------------------------------------------------------------

class Dataset:
    def __init__(self, metadata_path):
        super().__init__()

        if not metadata_path:
            raise Exception("No path to metadata given!")

        self.__metadata_path = metadata_path
        self.__load_dataset()

         # set starting point
        self.__current_file = self.__metadata_path

        # set starting index
        self.__index = self.__metadata[self.__metadata_path].get('last_idx') or 0

        # set current class
        self.__current_class = self.__metadata[self.__metadata_path].get('last_class') or 0

        # for safety
        self.get_item(self.__index);

    # -------------------------------------
    def __load_dataset(self):
        self.__metadata = {}
        self.__dataset = {}
        self.__classes = {}
        self.__calibrations = {}
        self.__sources = {}

        self.__size = 0

        self.__file_hierarchy = {
            self.__metadata_path: {
                'parent': None,
                'abspath': os.path.abspath(self.__metadata_path)
            }
        }
        self.__load_metadata( self.__metadata_path )

    # -------------------------------------
    def __load_metadata(self, metadata_path, parent="", absparent=""):
        """
        Load the metadata and recursively load all the embedded metadatas
        """
        
        metadata_path_abs = os.path.abspath(os.path.join(os.path.dirname(absparent), metadata_path))
        metadata_path_abs = self.__fix_path_separators(metadata_path_abs)

        if not os.path.exists(metadata_path_abs):
            print(f"not os.path.exists({metadata_path_abs})")
            return

        # add link to parent file and the absolute path for image loading and re-saving
        if metadata_path not in self.__file_hierarchy:
            self.__file_hierarchy[metadata_path] = {'parent': parent, 'abspath': metadata_path_abs}

        try:
            content = json.load( open(metadata_path_abs, "r") )
        except:
            log(f"Unable to parse the JSON. Check the file for errors!", msg_type=MessageType.ERROR)
            exit(1)
            return

        # the json content of all files
        self.__metadata[metadata_path] = content

        # the classes:
        self.__classes[metadata_path] = content.get('classes') or self.__classes.get(self.__metadata_path) or []

        # the calibrations
        calibration = content.get('calibration')

        # re-parse the numpy arrays:
        if calibration:
            for source, data in calibration.items():
                if not isinstance(data, dict):
                    calibration[source] = np.array(data)
                    continue
                for target, matrices in data.items():
                    if not isinstance(matrices, dict):
                        calibration[source][target] = np.array(matrices)
                        continue
                    for mat_name, mat in matrices.items():
                        calibration[source][target][mat_name] = np.array(mat)

        self.__calibrations[metadata_path] = calibration

        # load sources:
        self.__sources[metadata_path] = content.get('sources') or (self.__sources.get(self.__metadata_path) or ['default'])

        # actual list of files
        files = content.get('files')
        if files is None:
            return

        # split into actual proper files and embedded metadatas
        embedded_files = [file['metadata'] for file in files if 'metadata' in file]
        self.__file_hierarchy[metadata_path]['children'] = embedded_files

        files = [file for file in files if 'metadata' not in file]
        self.__dataset[metadata_path] = files

        # increase dataset size:
        self.__size += len(files)

        # run the same again for each embedded file:
        for file in embedded_files:
            # recursively call the `load_metadata` function for the embedded file
            self.__load_metadata(file, metadata_path, metadata_path_abs)

    # -------------------------------------
    # get current top-level metadata path:
    def get_metadata_path(self):
        return self.__metadata_path
    
    # -------------------------------------
    # traverse the full dataset file-struture in post-order (bottom to top)
    def __traverse(self, node=None, action=None, params=None):
        """
        Traverse the file hierarchy (depth-first)
        """

        if node is None:
            node = self.__metadata_path

        if node is None:
            return

        if 'children' not in self.__file_hierarchy[node]:
            return

        for child in self.__file_hierarchy[node]['children']:
            self.__traverse(child, action=action, params=params)

        # DO SOMETHING:
        if callable(action):
            if params is None:
                action(node)
            else:
                action(node, params)
                

    # -------------------------------------
    # save file
    def save_dataset(self, verbose=True):
        self.__traverse(action=self.__save_metadata, params=verbose)

    # -------------------------------------
    # save one file
    def __save_metadata(self, metadata_path, verbose=True):
        
        if verbose is None:
            verbose = True

        # update the metadata
        metadata = copy.deepcopy(self.__metadata[metadata_path])
        metadata['files'] = copy.deepcopy(self.__dataset[metadata_path])

        # if top:
        if self.__file_hierarchy[metadata_path]['parent'] is None:
            metadata['last_idx'] = self.__index
            if self.__current_class:
                data['last_class'] = int(self.__current_class)

        # re-add the embedded metadata files:
        for child in self.__file_hierarchy[metadata_path]['children']:
            metadata['files'].append({'metadata': child})

        ## save file
        #1. Create backup:
        save_path = self.__file_hierarchy[metadata_path]['abspath']
        backup_name = os.path.splitext(save_path)[0] + ".bak"

        # fix path separators
        save_path = self.__fix_path_separators(save_path)
        backup_name = self.__fix_path_separators(backup_name)

        shutil.copyfile(save_path, backup_name)

        #2. Try to save:
        try:
            #2a. save
            open(save_path, "w").write(json.dumps(metadata, indent=2, cls=NumpyEncoder))

            #2b. remove the backup if successful:
            if os.path.exists(backup_name):
                os.remove(backup_name)

            if verbose:
                log(f"Saved the metadata file '{save_path}'")
        except Exception:
            #3. restore the backup if error:
            if verbose:
                log(f"Failed to save the metadata file '{save_path}'", msg_type=MessageType.ERROR)
                traceback.print_exc()

            # restore backup:
            if os.path.exists(save_path):
                os.remove(save_path)
            if os.path.exists(backup_name):
                shutil.copyfile(backup_name, save_path)

    # -------------------------------------
    # simple extraction:
    def get_item(self, index):
        if index < 0 or index >= self.__size:
            return None

        self.__index = index
        local_index = index

        for metadata_path, files in self.__dataset.items():
            if local_index >= len(files):
                local_index -= len(files)
                continue
            else:
                self.__current_file = metadata_path
                break

        return self.__dataset[self.__current_file][local_index]

    # -------------------------------------
    # get the absolute path prefix for the current image
    def get_path_prefix(self):
        return os.path.dirname( self.__file_hierarchy[self.__current_file]['abspath'] )

    # -------------------------------------
    def get_image_path(self, image_path):
        full_path = os.path.join(self.get_path_prefix(), self.__fix_path_separators(image_path))
        return full_path

    def __fix_path_separators(self, path):
        new_path = path

         # if windows, check that there are no '/' slashes:
        if os.name == 'nt':
            new_path = new_path.replace('/', '\\')
        # if linux, replace every '\' with a '/' slach
        else:
            new_path = new_path.replace('\\', '/')

        return new_path

    # -------------------------------------
    # current file index
    def get_current_index(self):
        return np.clip(self.__index, 0, self.__size - 1)

    # -------------------------------------
    # current set of classes
    def get_classes(self):
        return self.__classes[self.__current_file]

    # -------------------------------------
    # returns a set of 'squished' class-IDs of the selected type
    def get_squished_classes(self, types=None):
        class_ids = {}

        if types is not None and not isinstance(types, list):
            types = [types]

        for class_set in self.__classes.values():
            current_classes = [c for c in class_set] if types is None else [c for c in class_set if c.get('type') in types]

            for c in current_classes:
                class_id = c.get('class_id')
                class_name = c.get('class_name')
                if class_name not in class_ids:
                    class_ids[class_name] = {'new_id': len(class_ids), 'old_id': class_id}

        return class_ids

    # -------------------------------------
    # current set of image sources
    def get_sources(self):
        return self.__sources[self.__current_file]

    # -------------------------------------
    # current set of calibrations
    def get_calibration(self):
        return self.__calibrations[self.__current_file]

    # -------------------------------------
    # finds a class based on parameter
    def __class_id_to_class(self, value, field):
        if value is None:
            return None

        c = [c for c in self.__classes[self.__current_file] if c.get(field) == value] or None

        if c is not None and len(c):
            return c[0]
        else:
            return None

    # -------------------------------------
    # split the data into training and testing/validation parts,
    # and return them as "{image: annotation}" dictionary
    def split_training_data(self, types=None, val_fraction=0.1, shuffle=True, max_val_size=None):
        class_ids = self.get_squished_classes(types)

        if types is not None and not isinstance(types, list):
            types = [types]

        img_files = []

        for idx in range(self.__size):
            image_data = self.get_item(idx).get('data') or []
            for src_idx in range(len(image_data)):
                current_image_data = image_data[src_idx]
                image_path = self.get_image_path(current_image_data.get('image'))
                if image_path is None:
                    continue
                annotations = unpack_annotation(current_image_data, self.get_classes(), unnormalize=False, rect_xywh2xyxy=False)
                labels = [obj for obj in annotations] if types is None else [obj for obj in annotations if obj['class_type'] in types]
                
                for obj in labels:
                    obj['new_id'] = class_ids[obj['class_name']]['new_id']

                img_file = {'path': image_path, 'labels': labels, 'img_format': current_image_data.get('image_format')}
                img_files.append(img_file)

        total_size = len(img_files)
        val_size = math.ceil(total_size * val_fraction)

        if max_val_size is not None and isinstance(max_val_size, int):
            val_size = np.clip(val_size, 0, max_val_size)

        indexes = np.arange(total_size)

        if shuffle:
            np.random.shuffle(indexes)

        val_part = sorted(indexes[-val_size:]) if val_fraction else []
        training_part = sorted(indexes[:-val_size]) if val_fraction else sorted(indexes)
        

        self.training_dataset = list(itemgetter(*training_part)(img_files)) if len(training_part) > 1 else [img_files[training_part[0]]] if len(training_part) else []
        self.validation_dataset = list(itemgetter(*val_part)(img_files)) if len(val_part) > 1 else [img_files[val_part[0]]] if len(val_part) else []

        return self.training_dataset, self.validation_dataset

    def get_training_dataset(self):
        if hasattr(self, 'training_dataset'):
            return self.training_dataset
        else:
            return None

    def get_validation_dataset(self):
        if hasattr(self, 'validation_dataset'):
            return self.validation_dataset
        else:
            return None

    # -------------------------------------
    # add annotation class_id to class number
    def add_annotation(self, idx, target_class, point_data, is_normalized=None, source=None, use_mapping=False, is_xyxy=False, force_uid=None):
        """
        idx [Integer]           The global index of the image, to which to add an annotation

        target_class [String]   The ID or Name of the annotated class

        point_data [Array]      The data to be added

        source [String, Int]    If using multisource, specify the name or the number of the image source, to which the annotation is added

        use_mapping [Boolean]   Try to map the coordinates to other sources, if calibration data is available

        is_normalized [Boolean] The provided coordinates ARE normalized to (0..1), otherwise the function does it for you

        is_xyxy [Boolean]       If it's a rectangle, the coordinates are given as TL (x1,y1) and BR (x2,y2) points, otherwise - (cX,cY,W,H)
        
        force_uid [String]      Overwrite the UUID
        """

        # get the image data (here, the correct CLASSES and CALIBRATIONS will also be chosen)
        image_data = self.get_item(idx)
        assert(image_data), "No such element index found in the Dataset"

        image_data = image_data.get('data') or []

        # if using multisource, and source not specified
        if len(image_data) > 1:
            assert(source is not None), "Target source not specified"

            if isinstance(source, str):
                source_name = source
            elif isinstance(source, int):
                source_name = image_data[np.clip(source, 0, len(image_data)-1)]['source']

            current_image_data = [d for d in image_data if d.get('source') == source_name] or None
            assert(current_image_data), "Image data with this source not found"
            current_image_data = current_image_data[0]
        else:
            current_image_data = image_data[0]
            source_name = current_image_data.get('source')

        # find the class
        current_class = self.__class_id_to_class(target_class, 'class_id') or self.__class_id_to_class(target_class, 'class_name')
        assert(current_class), "Class not found"

        # get class type, id and name
        class_type = current_class.get('type')
        class_id = current_class['class_id']
        class_name = current_class['class_name']

        # get calibrations if available:
        calibration = self.get_calibration()

        # transform point data to a proper array form
        points = np.asarray(point_data)

        # if rectangle and xywh:
        if class_type == 'rectangle' and not is_xyxy:
            points, z = points[:4], points[4:]
            z = z if len(z) else None
        else:
            z = None

        # add one dimension
        points = np.expand_dims(points, 0) if points.ndim == 1 else points

        # if rectangle and xywh:
        if class_type == 'rectangle' and not is_xyxy:
            points = xywh2xyxy(points)
            points = points.reshape((-1,2))
            if z is not None:
                z = np.repeat(z, points.shape[0]) if len(z) == 1 else z[:points.shape[0]]
                points = np.hstack([points, z.reshape((-1,1))])

         # try to infer whether the points are normalized or not:
        if is_normalized is None:
            # check if all X and Y coordinates are 0.0 <= .. <= 1.0:
            is_normalized = (np.all(points[:,:2] <= 1.0) and np.all(points[:,:2] >= 0.0))

        # generate a uuid:
        uid = generate_annotation_uuid(class_type) if force_uid is None else force_uid

        for target in [source_name] if not calibration or not use_mapping else calibration.keys():
            # copy points before mapping
            mapped_points = points.copy()

            # get the target image_data:
            target_image_data = [d for d in image_data if d.get('source') == target][0]

            # different source, needs mapping
            if target != source_name:
                mapped_points = map_coordinates(mapped_points, calibration, source_name, target, normalized=is_normalized)
            else:
                mapped_points = mapped_points[:, :2]

            # normalize if necessaryL
            if not is_normalized:
                mapped_points = self.__normalize_points(mapped_points, target_image_data)
                if mapped_points is None:
                    continue

            # if rectangle, convert back to [xywh]:
            if class_type == 'rectangle':
                #mapped_points = xyxy2xywh(np.clip(mapped_points.reshape((1, -1)), 0, 1))
                mapped_points = xyxy2xywh(mapped_points.reshape((1, -1)))

            # squeeze the points and turn Numpy into a list
            mapped_points = mapped_points.squeeze().tolist()

            # check if the annotation already exists there:
            target_image_data['annotation'] = target_image_data.get('annotation') or {}

            # check if an annotation for this particular class already exists
            target_image_data['annotation'][class_id] = target_image_data['annotation'].get(class_id) or []

            # finally, add the annotation:
            annotation = {
                'uid': uid,
                'object': mapped_points,
                'type': class_type,
                'created': get_annotation_timestamp()
            }

            # add reference to the source, from which this object was mapped
            if source_name is not None and target != source_name:
                annotation['mapped_from'] = source_name

            #print("Added annotation:\n", json.dumps(annotation, indent=2))
            target_image_data['annotation'][class_id].append(annotation)

    # -------------------------------------
    # normalize points depending on the target image size
    def __normalize_points(self, points, image_data):

        source_name = image_data.get('source')

        calibration = self.get_calibration()

        if calibration is None or source_name is None:
            img = cv2.imread(self.get_image_path(image_data['image']), -1)
            src_size = img.shape[:2][::-1]
        else:
            src_size = (calibration.get(source_name) or {}).get('imgSize')

        if src_size is not None:
            src_size = np.array(src_size)
            if points.shape[1] > 2:
                src_size = np.hstack([src_size, np.ones(points.shape[1] - 2)])
            # normalize:
            return points / src_size

        return None

    # -------------------------------------
    # return full size of the dataset
    def __len__(self):
        return self.__size

#----------------------------------------------------------------------------------------
def unpack_object(obj, image=None, unnormalize=True, rect_xywh2xyxy=True):
    pts = np.array(obj.get('object') or []).copy()

    # unsqueeze the points array
    pts = np.expand_dims(pts, 0) if pts.ndim == 1 else pts

    # un-normalize points [0..1, 0..1] -> [0..w, 0..h], and FLOAT -> INT32
    if image is not None and unnormalize:
        pts = (pts * np.resize(np.array(image.shape[:2][::-1]), pts.shape[1])).astype(np.int32)

    #!! get object type
    class_type = obj.get('type')

    # if it's a 'polygon' type
    if class_type == 'polygon':
        pts = pts.squeeze()

    # if it's a rectangle
    elif class_type == 'rectangle':
        if rect_xywh2xyxy:
            pts = xywh2xyxy(pts).squeeze()

    # if it's a point
    elif class_type == 'point':
        pts = pts.squeeze()

    return pts, class_type

#----------------------------------------------------------------------------------------
def unpack_annotation(image_data, classes=None, image=None, unnormalize=True, rect_xywh2xyxy=True):

    if isinstance(image_data, list):
        image_data = image_data[0]

    # check if there are annotations
    annotation = image_data.get('annotation') or {}

    annotation_objects = []

    #!! iterate over the classes in th annotation:
    for class_id, objects in annotation.items():

        class_idx = list_find(classes, lambda x: x['class_id'] == class_id, default=None) if classes is not None else None
        class_name = classes[class_idx]['class_name'] if classes is not None and class_idx is not None else class_id

        #!! get the objects of that class:
        for i, obj in enumerate(objects):

            #!! get the object (make a copy, to not mess up the original) and the type
            points, class_type = unpack_object(obj, image, unnormalize=unnormalize, rect_xywh2xyxy=rect_xywh2xyxy)

            annotation_object = {
                'object': points,
                'class_type': class_type,
                'class_id': class_id,
                'uid': obj.get('uid'),
                'class_name': class_name }

            annotation_objects.append(annotation_object)

    return annotation_objects

#----------------------------------------------------------------------------------------
def map_coordinates(source_points, calibration, source, target, z=1000, src_size=None, trg_size=None, normalized=True):

    if calibration is None:
        return None

    calibration_matrices = (calibration.get(source) or {}).get(target)
    if calibration_matrices is None:
        return

    if source_points.shape[1] < 3:
        # if no Z given, just set as 1 meter by default
        z = [z] * source_points.shape[0]
    else:
        z = source_points[:, -1].copy()

    r_matrix = np.array(calibration_matrices.get("Rotation"))
    t_matrix = np.array(calibration_matrices.get("Translation"))

    k_matrix_1 = np.zeros((3,3))    # source K-matrix is 3x3, it will be inverted
    k_matrix_2 = np.zeros((3,4))    # target K-matrix is 3x4

    k_matrix_1[:3, :3] = calibration.get(source).get('K')
    k_matrix_2[:3, :3] = calibration.get(target).get('K')

    # image shapes
    if normalized:
        shape_1 = np.array(src_size if src_size else calibration.get(source).get('imgSize'))
        shape_2 = np.array(trg_size if trg_size else calibration.get(target).get('imgSize'))

        if np.isnan(shape_1).any() or np.isnan(shape_2).any():
            return None

    # check that all matrices are correct
    if r_matrix is None or t_matrix is None or np.isnan(k_matrix_1).any() or np.isnan(k_matrix_2).any():
        return None

    # build the transformation matrix
    trans_matrix = np.eye(4,4)
    trans_matrix[:3, :3] = r_matrix  # 3 x 3
    trans_matrix[:3, -1:] = t_matrix # 3 x 1

    # invert the source K-matrix
    k_matrix_1_i = np.linalg.inv(k_matrix_1)

    target_points = source_points[:, :2].copy()

    # for each point:
    for i, pt in enumerate(target_points):
        # un-normalize the point to full image size
        pt1 = (pt[:2] * shape_1) if normalized else pt[:2]

        # pad to (x, y, 1) and convert to real-world coordinates
        world_3D_coord = np.dot(k_matrix_1_i, np.hstack([pt1, 1])) * z[i]

        # pad the world coordinate to (X, Y, Z, 1), and transpose
        world_3D_coord = np.hstack([world_3D_coord, 1]).reshape((-1, 1))

        # transform, convert to target-camera pixel-coordinates, divide by Z
        new_pt = np.matmul(k_matrix_2, np.matmul(trans_matrix, world_3D_coord)) / world_3D_coord[2]

        # divide again by the (z), crop only the (x, y):
        new_pt /= new_pt[2]
        new_pt = new_pt[:2].squeeze()

        # print(f"Mapped point from {pt1} ({source}) to {new_pt} ({target}) with depth: {z[i]}")

        # normalize the coordinate to [0..1, 0..1]
        if normalized:
            new_pt /= shape_2

        # save the coordinate in-place:
        pt[:2] = new_pt

    return target_points