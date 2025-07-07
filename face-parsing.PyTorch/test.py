#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))



def get_voc_palette(n_cls):
    """
    Generate a VOC-style palette for n_cls classes.
    Returns a flat list of length n_cls*3.
    """
    palette = []
    for i in range(n_cls):
        lab = i
        r = g = b = 0
        for j in range(8):
            r |= ((lab >> 0) & 1) << (7 - j)
            g |= ((lab >> 1) & 1) << (7 - j)
            b |= ((lab >> 2) & 1) << (7 - j)
            lab >>= 3
        palette.extend([r, g, b])
    return palette

def segment_faces_in_folder(input_dir, output_dir,
                            checkpoint='79999_iter.pth',
                            img_size=(512,512),
                            n_classes=19):
    # create output folder if needed
    os.makedirs(output_dir, exist_ok=True)

    # load network
    net = BiSeNet(n_classes=n_classes).cuda()
    net.load_state_dict(torch.load(checkpoint))
    net.eval()

    # preprocessing
    to_tensor = transforms.Compose([
        transforms.Resize(img_size, Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # build palette once
    palette = get_voc_palette(n_classes)

    with torch.no_grad():
        for fname in os.listdir(input_dir):
            in_path  = osp.join(input_dir, fname)
            out_path = osp.join(output_dir, fname)

            # load & preprocess
            img = Image.open(in_path).convert('RGB')
            tensor = to_tensor(img).unsqueeze(0).cuda()

            # forward & get class‐ids
            out = net(tensor)
            if isinstance(out, (list, tuple)):
                out = out[-1]            # final head
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # create a paletted image and save
            mask = Image.fromarray(parsing.astype(np.uint8), mode='P')
            mask.putpalette(palette)
            mask.save(out_path)

if __name__ == "__main__":
    evaluate(dspth='/home/zll/data/CelebAMask-HQ/test-img', cp='79999_iter.pth')


