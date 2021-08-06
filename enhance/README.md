# 

## 修改内容
- 增加scripts文件夹
- 增加file_utils.py和image_utils.py工具
- 增加支持VOC数据格式训练：parser_voc.py和voc_datasets.py,需求修改`train.py`和`utils/datasets.py`
```python
        # (1)train.py中create_dataloader中增加参数:data_dict
        # (2)utils/datasets.py选择不同的数据类型：data_type
        if "data_type" in data_dict and data_dict["data_type"] == "voc":
            from utils import voc_datasets
            dataset = voc_datasets.LoadVOCImagesAndLabels(path, imgsz, batch_size,
                                                          augment=augment,  # augment images
                                                          hyp=hyp,  # augmentation hyperparameters
                                                          rect=rect,  # rectangular training
                                                          cache_images=cache,
                                                          single_cls=single_cls,
                                                          stride=int(stride),
                                                          pad=pad,
                                                          image_weights=image_weights,
                                                          prefix=prefix,
                                                          names=data_dict["names"])
        else:
            dataset = LoadImagesAndLabels
```  
- dataset中显示img, labels
```python

    def show_cxcywh(self, img, labels):
        height, width, d = img.shape
        rects = []
        classes = []
        for item in labels:
            c, cx, cy, w, h = item
            rect = [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]
            rects.append(rect)
            classes.append(c)
        image_utils.show_image_rects_text("image", img, rects, classes)

```