# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/coco/zero-shot/instances_train2017_seen_2.json')
    parser.add_argument('--cat_path', default='datasets/coco/annotations/instances_val2017.json')
    args = parser.parse_args()
    print('Loading', args.cat_path)
    unseen_set = [
        "umbrella",
        "cow",
        "cup",
        "bus",
        "keyboard",
        "skateboard",
        "dog",
        "couch",
        "tie",
        "snowboard",
        "sink",
        "elephant",
        "cake",
        "scissors",
        "airplane",
        "cat",
        "knife"
    ]
    seen_set = [
        "toilet",
        "bicycle",
        "apple",
        "train",
        "laptop",
        "carrot",
        "motorcycle",
        "oven",
        "chair",
        "mouse",
        "boat",
        "kite",
        "sheep",
        "horse",
        "sandwich",
        "clock",
        "tv",
        "backpack",
        "toaster",
        "bowl",
        "microwave",
        "bench",
        "book",
        "orange",
        "bird",
        "pizza",
        "fork",
        "frisbee",
        "bear",
        "vase",
        "toothbrush",
        "spoon",
        "giraffe",
        "handbag",
        "broccoli",
        "refrigerator",
        "remote",
        "surfboard",
        "car",
        "bed",
        "banana",
        "donut",
        "skis",
        "person",
        "truck",
        "bottle",
        "suitcase",
        "zebra",
        "background"
    ]

    cates_80 = json.load(open(args.cat_path, 'r'))['categories']
    cates_65 = []
    id_map = {}
    new_id = 1
    for cate in cates_80:
        if cate['name'] in seen_set or cate['name'] in unseen_set:
                id_map[cate['id']] = new_id
                cate['id'] = new_id
                new_id = new_id + 1
                cates_65.append(cate)

    print('Loading', args.data_path)
    data = json.load(open(args.data_path, 'r'))
    data['categories'] = cates_65

    for anno in data['annotations']:
        anno['category_id'] = id_map[anno['category_id']]

    out_path = args.data_path[:-5] + '_del.json'
    print('Saving to', out_path)
    json.dump(data, open(out_path, 'w'))
