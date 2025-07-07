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

def segment_faces_in_folder(input_dir,
                            output_dir,
                            checkpoint='79999_iter.pth',
                            img_size=(512, 512),
                            n_classes=19,
                            face_classes=tuple(range(1, 10))):
    """
    Runs BiSeNet on every image in input_dir, then keeps only the face
    (class IDs 1–9) and saves an RGB image with a black background.
    """
    os.makedirs(output_dir, exist_ok=True)

    # load network
    net = BiSeNet(n_classes=n_classes).cuda()
    net.load_state_dict(torch.load(checkpoint))
    net.eval()

    # preprocessing: resize→tensor→normalize
    preprocess = transforms.Compose([
        transforms.Resize(img_size, Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        for fname in os.listdir(input_dir):
            in_path = osp.join(input_dir, fname)
            out_path = osp.join(output_dir, fname)

            # load original
            orig = Image.open(in_path).convert('RGB')
            orig_w, orig_h = orig.size

            # prepare network input
            resized = orig.resize(img_size, Image.BILINEAR)
            tensor = preprocess(resized).unsqueeze(0).cuda()

            # forward pass
            out = net(tensor)
            # if multiple heads, take the last
            if isinstance(out, (list, tuple)):
                out = out[-1]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)  # H×W class IDs

            # build binary face mask (512×512)
            mask512 = np.isin(parsing, face_classes).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask512, mode='L')

            # resize mask back to original size
            mask_full = mask_img.resize((orig_w, orig_h), Image.NEAREST)
            mask_np = np.array(mask_full)[:, :, None]  # H×W×1

            # apply mask: multiply each channel by (mask>0)
            orig_np = np.array(orig)
            face_np = (orig_np * (mask_np > 0)).astype(np.uint8)

            # save result
            Image.fromarray(face_np).save(out_path)

if __name__ == "__main__":
    evaluate(dspth='/home/zll/data/CelebAMask-HQ/test-img', cp='79999_iter.pth')


