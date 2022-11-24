import os
name="lvis_softtagc_allimg_s100_max9w_lr00002_p16_w01"
for root, dirs, files in os.walk('output/%s/' % name):
    for model in files:
        if model.endswith('.pth'):
            #print(os.path.join(root, name))
            os.system('python3 train_net.py --num-gpus 8 --config-file configs/Detic_LbaseCCcap_CLIP_R5021k_640b64_4x_ft4x_caption.yaml --eval-only OUTPUT_DIR output/%s MODEL.WEIGHTS output/%s/%s' % (name, name, model))
