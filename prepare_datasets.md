# Prepare datasets

The training of our work is on two benchmark datasets: [COCO](https://cocodataset.org/) and [COCOCaption](https://cocodataset.org/), [LVIS](https://www.lvisdataset.org/) and [Conceptual Caption (CC3M)](https://ai.google.com/research/ConceptualCaptions/).
Please orignize the datasets as following.
```
$VLDet_ROOT/datasets/
    lvis/
    coco/
    cc3m/
```

Please follow the following instruction to pre-process individual datasets.

### COCO and COCO Caption

First, download COCO data place them in the following way:

```
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
```

We first follow [OVR-CNN](https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb) to create the open-vocabulary COCO split. The converted files should be like:

```
coco/
    zero-shot/
        instances_train2017_seen_2.json
        instances_val2017_all_2.json
```

We further preprocess the annotation format for easier evaluation:

```
python tools/get_coco_zeroshot.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
python tools/get_coco_zeroshot.py --data_path datasets/coco/zero-shot/instances_val2017_all_2.json
```

And process the category infomation: 

```
python tools/get_lvis_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_del.json
```

Next, we prepare the class and concept embedding following RegionCLIP. Download the embedding file from [here](https://drive.google.com/drive/folders/1_HKaLSyA9fIjTS0BXxT15NRac6uRoyIj) or generate it with tools from [RegionCLIP](https://github.com/microsoft/RegionCLIP/tree/9fd374015db384bc0548b4af85446b90e13d2ae1#extract-concept-features). The files should be like: 

```
coco/
    VLDet/
        coco_65_cls_emb.pth
        coco_65_concepts.txt
        coco_nouns_4764_emb.pth
        coco_nouns_4764.txt
```

Then preprocess the COCO caption data with the predefined concepts:

```
python tools/get_tags_for_VLDet_concepts.py --cc_ann datasets/coco/annotations/captions_train2017.json --allcaps --cat_path datasets/coco/VLDet/coco_nouns_4764.txt --convert_caption --out_path datasets/coco/VLDet/nouns_captions_train2017_4764tags_allcaps.json
``` 

This creates `datasets/coco/VLDet/nouns_captions_train2017_4764tags_allcaps.json`.

### LVIS

First, download LVIS data place them in the following way:

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
```
Next, prepare the open-vocabulary LVIS training set using 

```
python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
```

This will generate `datasets/lvis/lvis_v1_train_norare.json`.

`lvis_v1_train_cat_info.json` is used by the Federated loss.
This is created by 
~~~
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train.json
~~~

### Conceptual Caption

Download the dataset from [this](https://ai.google.com/research/ConceptualCaptions/download) page and place them as:
```
cc3m/
    GCC-training.tsv
```

Run the following command to download the images and convert the annotations to LVIS format (Note: download images takes long).

~~~
python tools/download_cc.py --ann datasets/cc3m/GCC-training.tsv --save_image_path datasets/cc3m/training/ --out_path datasets/cc3m/train_image_info_tags.json
~~~

This creates `datasets/cc3m/train_image_info_tags.json`.

Next, we prepare the class and concept embedding following RegionCLIP. Download the embedding file from [here](https://drive.google.com/drive/folders/1_HKaLSyA9fIjTS0BXxT15NRac6uRoyIj) or generate it with tools from [RegionCLIP](https://github.com/microsoft/RegionCLIP/tree/9fd374015db384bc0548b4af85446b90e13d2ae1#extract-concept-features). The files should be like: 

```
cc3m/
    VLDet/
        googlecc_nouns_6250_emb.pth
        googlecc_nouns_6250.txt
        lvis_1203_cls_emb.pth
```

Run the following command to convert the annotations to LVIS format:

```
python tools/get_tags_for_VLDet_concepts.py
```

This creates `datasets/cc3m/VLDet/nouns_train_image_info_6250tags.json`
