# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""
import os
import xmltodict
import numpy as np
import cv2
import glob
import random
import numbers
from tqdm import tqdm


class Dataset(object):
    """
    from torch.utils.data import DataLoader, ConcatDataset
    """

    def __init__(self, **kwargs):
        self.image_id = []

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def read_files(filename, *args):
        """
        :param filename:
        :return:
        """
        image_id = []
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip().split(" ")[0]
                image_id.append(line.rstrip())
        return image_id


class VOCDataset(Dataset):

    def __init__(self,
                 filename=None,
                 data_root=None,
                 anno_dir=None,
                 image_dir=None,
                 class_names=None,
                 transform=None,
                 target_transform=None,
                 color_space="RGB",
                 keep_difficult=False,
                 shuffle=False,
                 check=False):
        """
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :param transform:
        :param target_transform:
        :param color_space:
        :param keep_difficult:
        :param shuffle:
        """
        super(VOCDataset, self).__init__()
        self.class_names, self.class_dict = self.parser_classes(class_names)
        parser = self.parser_paths(filename, data_root, anno_dir, image_dir)
        self.data_root, self.anno_dir, self.image_dir, self.image_id = parser
        self.postfix = self.get_image_postfix(self.image_dir, self.image_id)
        self.transform = transform
        self.target_transform = target_transform
        self.color_space = color_space
        self.keep_difficult = keep_difficult
        if check:
            self.image_id = self.checking(self.image_id)
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_id)
        self.num_images = len(self.image_id)
        self.classes = list(self.class_dict.values())
        self.num_classes = max(list(self.class_dict.values())) + 1 if self.class_dict else None
        print("class_dict:{}".format(self.class_dict))
        print("image id:{}".format(len(self.image_id)))

    def get_image_postfix(self, image_dir, image_id):
        """
        获得图像文件后缀名
        :param image_dir:
        :return:
        """
        if "." in image_id[0]:
            postfix = ""
        else:
            image_list = glob.glob(os.path.join(image_dir, "*"))
            postfix = os.path.basename(image_list[0]).split(".")[1]
        return postfix

    def __get_image_anno_file(self, image_dir, anno_dir, image_id: str, img_postfix):
        """
        :param image_dir:
        :param anno_dir:
        :param image_id:
        :param img_postfix:
        :return:
        """
        if not img_postfix and "." in image_id:
            image_id, img_postfix = image_id.split(".")
        image_file = os.path.join(image_dir, "{}.{}".format(image_id, img_postfix))
        annotation_file = os.path.join(anno_dir, "{}.xml".format(image_id))
        return image_file, annotation_file

    def checking(self, image_ids: list, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        dst_ids = []
        # image_ids = image_ids[:100]
        # image_ids = image_ids[100:]
        for image_id in tqdm(image_ids):
            image_file, annotation_file = self.get_image_anno_file(image_id)
            if not os.path.exists(annotation_file):
                continue
            if not os.path.exists(image_file):
                continue
            objects = self.get_annotation(annotation_file)
            bboxes, labels, is_difficult = objects["bboxes"], objects["labels"], objects["is_difficult"]
            if not self.keep_difficult:
                bboxes = bboxes[is_difficult == 0]
                # labels = labels[is_difficult == 0]
            if ignore_empty and (len(bboxes) == 0 or len(labels) == 0):
                print("empty annotation:{}".format(annotation_file))
                continue
            dst_ids.append(image_id)
        print("have nums image:{},legal image:{}".format(len(image_ids), len(dst_ids)))
        return dst_ids

    def parser_classes(self, class_names):
        """
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        :param class_names:
                    str : class file
                    list: ["face","person"]
                    dict: 可以自定义label的id{'BACKGROUND': 0, 'person': 1, 'person_up': 1, 'person_down': 1}
        :return:
        """
        if isinstance(class_names, str):
            class_names = super().read_files(class_names)
        if isinstance(class_names, list):
            class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        elif isinstance(class_names, dict):
            class_dict = class_names
        else:
            class_dict = None
        return class_names, class_dict

    def parser_paths(self, filenames=None, data_root=None, anno_dir=None, image_dir=None):
        """
        :param filenames:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :return:
        """
        if isinstance(data_root, str):
            anno_dir = os.path.join(data_root, "Annotations") if not anno_dir else anno_dir
            image_dir = os.path.join(data_root, "JPEGImages") if not image_dir else image_dir
        if isinstance(filenames, str):
            data_root = os.path.dirname(filenames)
        image_id = self.read_files(filenames, anno_dir)
        if not anno_dir:
            anno_dir = os.path.join(data_root, "Annotations")
        if not image_dir:
            image_dir = os.path.join(data_root, "JPEGImages")
        return data_root, anno_dir, image_dir, image_id

    def crop_image(self, image, bbox):
        """
        :param image:
        :param bbox:
        :return:
        """
        # bboxes = image_processing.extend_bboxes([bbox], scale=[1.5, 1.5])
        # bboxes = image_processing.extend_bboxes([bbox], scale=[1.2, 1.2])
        bboxes = image_processing.extend_bboxes([bbox], scale=[1.3, 1.3])
        images = image_processing.get_bboxes_crop_padding(image, bboxes)
        return images, bboxes

    def convert_target(self, boxes, labels):
        annotations = []
        for i in range(len(boxes)):
            bbox = boxes[i, :].tolist()
            label = labels[i].tolist()
            anno = list()
            anno += bbox
            anno += [label]
            assert len(anno) == 5
            annotations.append(anno)
        target = np.array(annotations)
        return target

    def __getitem__(self, index):
        """
        :param index: int or str
        :return:rgb_image
        """
        image_id = self.index2id(index)
        image_file, annotation_file = self.get_image_anno_file(image_id)
        objects = self.get_annotation(annotation_file)
        bboxes, labels, is_difficult = objects["bboxes"], objects["labels"], objects["is_difficult"]
        image = self.read_image(image_file, color_space=self.color_space)
        if not self.keep_difficult:
            index = is_difficult == 0
            bboxes = bboxes[index]
            labels = labels[index]
            # landms = landms[index]
        if self.transform:
            image, bboxes, labels = self.transform(image, bboxes, labels)
        num_boxes = len(bboxes)
        if self.target_transform:
            bboxes, labels = self.target_transform(bboxes, labels)  # torch.Size([29952, 4]),torch.Size([29952])

        target = self.convert_target(bboxes, labels)
        if num_boxes == 0:
            index = int(random.uniform(0, len(self)))
            return self.__getitem__(index)
        # return image, bboxes, labels
        return image, target

    def get_image_anno_file(self, index):
        """
        :param index:
        :return:
        """
        image_id = self.index2id(index)
        image_file, annotation_file = self.__get_image_anno_file(self.image_dir, self.anno_dir, image_id, self.postfix)
        return image_file, annotation_file

    def index2id(self, index):
        """
        :param index: int or str
        :return:
        """
        if isinstance(index, numbers.Number):
            image_id = self.image_id[index]
        else:
            image_id = index
        return image_id

    def __len__(self):
        return len(self.image_id)

    def check_bbox(self, width, height, bbox):
        xmin, ymin, xmax, ymax = bbox
        sw = (xmax - xmin) / width
        sh = (ymax - ymin) / height
        ok = True
        if sw < 0 or sw > 1:
            ok = False
        elif sh < 0 or sh > 1:
            ok = False
        return ok

    def get_annotation(self, xml_file):
        """
        :param xml_file:
        :param class_dict: class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        :return:
        """
        try:
            content = self.read_xml2json(xml_file)
            annotation = content["annotation"]
            # get image shape
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            depth = int(annotation["size"]["depth"])
            filename = annotation["filename"]
            objects = annotation["object"]
        except Exception as e:
            print("illegal annotation:{}".format(xml_file))
            objects = []
        objects_list = []
        if not isinstance(objects, list):
            objects = [objects]
        for object in objects:
            name = str(object["name"]).lower()
            if self.class_names and name not in self.class_names:
                continue
            difficult = int(object["difficult"])
            xmin = float(object["bndbox"]["xmin"])
            xmax = float(object["bndbox"]["xmax"])
            ymin = float(object["bndbox"]["ymin"])
            ymax = float(object["bndbox"]["ymax"])
            # rect = [xmin, ymin, xmax - xmin, ymax - ymin]
            bbox = [xmin, ymin, xmax, ymax]
            if not self.check_bbox(width, height, bbox):
                # print("illegal bbox:{}".format(xml_file))
                continue
            item = {}
            item["bbox"] = bbox
            item["difficult"] = difficult
            if self.class_dict:
                name = self.class_dict[name]
            item["name"] = name
            objects_list.append(item)
        bboxes, labels, is_difficult = self.get_objects_items(objects_list)
        objects = {"bboxes": bboxes,
                   "labels": labels,
                   "is_difficult": is_difficult,
                   "width": width,
                   "height": height,
                   }
        return objects

    def get_objects_items(self, objects_list):
        """
        :param objects_list:
        :return:
        """
        bboxes = []
        labels = []
        is_difficult = []
        for item in objects_list:
            bboxes.append(item["bbox"])
            labels.append(item['name'])
            is_difficult.append(item['difficult'])
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels)  # for string
        # labels = np.array(labels, dtype=np.int64)  # for int64
        # labels = np.asarray(labels).reshape(-1, 1)
        is_difficult = np.array(is_difficult, dtype=np.uint8)
        return bboxes, labels, is_difficult

    @staticmethod
    def read_files(filename, *args):
        """
        :param filename:
        :return:
        """
        if not filename:  # if None
            assert args
            anno_list = []
            for a in args:
                anno_list += file_processing.get_files(a, postfix=["*.xml"])
            image_id = VOCDataset.get_files_id(anno_list)
        elif isinstance(filename, list):
            image_id = filename
        elif isinstance(filename, str):
            # image_id = super().read_files(filename)
            image_id = Dataset.read_files(filename)
        else:
            image_id = None
            assert Exception("Error:{}".format(filename))
        return image_id

    @staticmethod
    def get_files_id(file_list):
        """
        :param file_list:
        :return:
        """
        image_idx = []
        for path in file_list:
            basename = os.path.basename(path)
            id = basename.split(".")[0]
            image_idx.append(id)
        return image_idx

    @staticmethod
    def read_xml2json(xml_file):
        """
        import xmltodict
        :param xml_file:
        :return:
        """
        with open(xml_file, encoding='utf-8') as fd:  # 将XML文件装载到dict里面
            content = xmltodict.parse(fd.read())
        return content

    def read_image(self, image_file, color_space="RGB"):
        """
        :param image_file:
        :param color_space:
        :return:
        """
        image = cv2.imread(str(image_file))
        if color_space.lower() == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class ConcatDataset(Dataset):
    """ Concat Dataset """

    def __init__(self, datasets, shuffle=False):
        """
        import torch.utils.data as torch_utils
        voc1 = PolygonParser(filename1)
        voc2 = PolygonParser(filename2)
        voc=torch_utils.ConcatDataset([voc1, voc2])
        ====================================
        :param datasets:
        :param shuffle:
        """
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'dataset should not be an empty iterable'
        # super(ConcatDataset, self).__init__()
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.image_id = []
        self.dataset = datasets
        self.shuffle = shuffle
        for dataset_id, dataset in enumerate(self.dataset):
            image_id = dataset.image_id
            image_id = self.add_dataset_id(image_id, dataset_id)
            self.image_id += image_id
            self.classes = dataset.classes
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_id)

    def add_dataset_id(self, image_id, dataset_id):
        """
        :param image_id:
        :param dataset_id:
        :return:
        """
        out_image_id = []
        for id in image_id:
            out_image_id.append({"dataset_id": dataset_id, "image_id": id})
        return out_image_id

    def __getitem__(self, index):
        """
        :param index: int
        :return:
        """
        dataset_id = self.image_id[index]["dataset_id"]
        image_id = self.image_id[index]["image_id"]
        dataset = self.dataset[dataset_id]
        # print(dataset.data_root, image_id)
        data = dataset.__getitem__(image_id)
        return data

    def get_image_anno_file(self, index):
        dataset_id = self.image_id[index]["dataset_id"]
        image_id = self.image_id[index]["image_id"]
        return self.dataset[dataset_id].get_image_anno_file(image_id)

    def get_annotation(self, xml_file):
        return self.dataset[0].get_annotation(xml_file)

    def read_image(self, image_file):
        return self.dataset[0].read_image(image_file, color_space=self.dataset[0].color_space)

    def __len__(self):
        return len(self.image_id)


def VOCDatasets(filenames=None,
                data_root=None,
                anno_dir=None,
                image_dir=None,
                class_names=None,
                transform=None,
                color_space="RGB",
                keep_difficult=False,
                shuffle=False,
                check=False):
    """
    :param filenames:
    :param data_root:
    :param anno_dir:
    :param image_dir:
    :param class_names:
    :param transform:
    :param color_space:
    :param keep_difficult:
    :param shuffle:
    :param check:
    :return:
    """
    if not isinstance(filenames, list) and os.path.isfile(filenames):
        filenames = [filenames]
    datas = []
    for filename in filenames:
        data = VOCDataset(filename=filename,
                          data_root=data_root,
                          anno_dir=anno_dir,
                          image_dir=image_dir,
                          class_names=class_names,
                          transform=transform,
                          color_space=color_space,
                          keep_difficult=keep_difficult,
                          shuffle=shuffle,
                          check=check)
        datas.append(data)
    voc = ConcatDataset(datas, shuffle=shuffle)
    return voc


def show_boxes_image(image, bboxes, labels, normal=False, transpose=False):
    """
    :param image:
    :param targets_t:
                bboxes = targets[idx][:, :4].data
                keypoints = targets[idx][:, 4:14].data
                labels = targets[idx][:, -1].data
    :return:
    """
    import numpy as np
    from utils import image_processing
    image = np.asarray(image)
    bboxes = np.asarray(bboxes)
    labels = np.asarray(labels)
    print("image:{}".format(image.shape))
    print("bboxes:{}".format(bboxes))
    print("labels:{}".format(labels))
    if transpose:
        image = image_processing.untranspose(image)
    h, w, _ = image.shape
    landms_scale = np.asarray([w, h] * 5)
    bboxes_scale = np.asarray([w, h] * 2)
    if normal:
        bboxes = bboxes * bboxes_scale
    # image = image_processing.untranspose(image)
    # image = image_processing.convert_color_space(image, colorSpace="RGB")
    image = image_processing.draw_image_bboxes_text(image, bboxes, labels)
    image_processing.cv_show_image("image", image, waitKey=0)
    print("===" * 10)


if __name__ == "__main__":
    from utils import image_processing, file_processing
    from modules.transforms import data_transforms

    # from models.transforms import data_transforms

    isshow = True
    # data_root = "/home/dm/panjinquan3/dataset/MPII/"
    # data_root = "/media/dm/dm2/git/python-learning-notes/dataset/Test_Voc"
    # anno_dir = '/home/dm/panjinquan3/dataset/finger/finger/Annotations'
    # image_dir = '/home/dm/panjinquan3/dataset/finger/finger/JPEGImages'
    # data_root = "/home/dm/panjinquan3/dataset/Character/gimage_v1/"
    # data_root = "/home/dm/panjinquan3/dataset/finger/finger_v5/"
    # data_root = '/home/dm/panjinquan3/dataset/Character/gimage_v1/'
    # data_root = "/home/dm/data3/dataset/face_person/SMTC/"
    data_root = "/home/dm/data3/dataset/face_person/MPII/"
    # data_root = "/home/dm/data3/dataset/face_person/COCO/VOC/"
    # data_root = "/home/dm/data3/dataset/card_datasets/yolo_det/CardData4det/"
    image_dir = data_root + "JPEGImages"
    anno_dir = data_root + "Annotations"
    filenames = data_root + "trainval.txt"
    # class_names = ["face", "person"]
    class_names = ["person"]
    # class_names = ["card","1"]
    # anno_dir = data_root + '/Annotations'
    shuffle = False
    # class_names = ["face", "person"]
    # class_names = None
    # class_names = {"circle": 0, "hook": 1, "slash": 2, "underline": 3}
    # anno_list = file_processing.get_files_list(anno_dir, postfix=["*.xml"])
    # image_id_list = file_processing.get_files_id(anno_list)
    size = [320, 320]
    transform = data_transforms.TrainTransform(size, mean=0.0, std=1.0, norm=True)
    # transform = data_transforms.DemoTransform(size, mean=0.0, std=1.0)
    voc = VOCDataset(filename=filenames,
                     data_root=None,
                     anno_dir=anno_dir,
                     image_dir=image_dir,
                     class_names=class_names,
                     transform=transform,
                     check=False)
    voc = ConcatDataset([voc, voc])
    # voc = torch_utils.ConcatDataset([voc, voc])
    print("have num:{}".format(len(voc)))
    for i in range(len(voc)):
        print(i)
        image, target = voc.__getitem__(i)
        bboxes, labels = target[:, 0:4], target[:, 4:5]
        show_boxes_image(image, bboxes, labels, normal=True, transpose=True)
