# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
from collections import defaultdict
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
from nltk import word_tokenize, pos_tag, ne_chunk
import re


def map_name(x):
    x = x.replace('_', ' ')
    if '(' in x:
        x = x[:x.find('(')]
    return x.lower().strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cc_ann', default='datasets/cc3m/train_image_info_tags.json')
    parser.add_argument('--out_path', default='datasets/cc3m/VLDet/nouns_train_image_info_6250tags.json')
    parser.add_argument('--keep_images', action='store_true')
    parser.add_argument('--allcaps', action='store_true')
    parser.add_argument('--cat_path', default='datasets/cc3m/VLDet/googlecc_nouns_6250.txt')
    parser.add_argument('--convert_caption', action='store_true')
    # parser.add_argument('--lvis_ann', default='datasets/lvis/lvis_v1_val.json')
    args = parser.parse_args()

    # lvis_data = json.load(open(args.lvis_ann, 'r'))
    cc_data = json.load(open(args.cc_ann, 'r'))
    if args.convert_caption:
        num_caps = 0
        caps = defaultdict(list)
        for x in cc_data['annotations']:
            caps[x['image_id']].append(x['caption'])
        for x in cc_data['images']:
            x['captions'] = caps[x['id']]
            num_caps += len(x['captions'])
        print('# captions', num_caps)

    if args.cat_path != '':
        print('Loading', args.cat_path)
        ff = open(args.cat_path)
        cats = []
        line = ff.readline()
        ii = 1
        while line:
            concept = line.split(',')[0]
            cats.append({'id': ii, 'name': concept, 'synonyms': [concept], 'frequency': 'f', 'supercategory': concept} )
            ii = ii+1
            line = ff.readline()
        ff.close()
        cc_data['categories'] = cats
        
        
    id2cat = {x['id']: x for x in cc_data['categories']}
    class_count = {x['id']: 0 for x in cc_data['categories']}
    class_data = {x['id']: [' ' + map_name(xx) + ' ' for xx in x['synonyms']] \
            for x in cc_data['categories']}
    num_examples = 5
    examples = {x['id']: [] for x in cc_data['categories']}

    print('class_data', class_data)

    images = []
    for i, x in enumerate(cc_data['images']):
        if i % 10000 == 0:
            print(i, len(cc_data['images']))
        new_caption = ' '
        if args.allcaps:
            new_caption = (' '.join(x['captions'])).lower()
        else:
            caption = x['captions'][0].lower()
            pos = pos_tag(word_tokenize(caption))
            new_caption = ' '
            for word, tag in pos:
                if re.match(r"NN.*", tag):
                    new_caption = new_caption + word + ' '

        x['pos_category_ids'] = []
        for cat_id, cat_names in class_data.items():
            find = False
            for c in cat_names:
                if c in new_caption or new_caption.startswith(c[1:]) \
                    or new_caption.endswith(c[:-1]):
                    find = True
                    break
            if find:
                x['pos_category_ids'].append(cat_id)
                class_count[cat_id] += 1
        if len(x['pos_category_ids']) > 0 or args.keep_images:
            images.append(x)

    zero_class = []
    for cat_id, count in class_count.items():
        print(id2cat[cat_id]['name'], count, end=', ')
        if count == 0:
            zero_class.append(id2cat[cat_id])
    print('==')
    print('zero class', zero_class)

    # for freq in ['r', 'c', 'f']:
    #     print('#cats', freq, len([x for x in cc_data['categories'] \
    #         if x['frequency'] == freq] and class_count[x['id']] > 0))

    for freq in ['r', 'c', 'f']:
        print('#Images', freq, sum([v for k, v in class_count.items() \
        if id2cat[k]['frequency'] == freq]))

    try:
        out_data = {'images': images, 'categories': cc_data['categories'], \
            'annotations': []}
        for k, v in out_data.items():
            print(k, len(v))
        if args.keep_images and not args.out_path.endswith('_full.json'):
            args.out_path = args.out_path[:-5] + '_full.json'
        print('Writing to', args.out_path)
        json.dump(out_data, open(args.out_path, 'w'))
    except:
        pass
