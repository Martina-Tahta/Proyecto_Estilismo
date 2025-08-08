import sys, torch, numpy as np
from torchvision import transforms
from PIL import Image
import os
import torch
import pandas as pd


# optional: auto-download with gdown if missing
try:
    import gdown
except ImportError:
    gdown = None
    
sys.path.insert(0, "face-parsing.PyTorch")       # relative path to the clone
from model import BiSeNet                         # now import works

# git clone --depth 1 https://github.com/zllrunning/face-parsing.PyTorch

# ── 3.2  one-liner to get the pretrained weights ────────────────────────────
def load_parser(device="cpu"):
    """
    Loads BiSeNet with CelebAMask-HQ weights.  
    If the .pth isn't in your clone, tries to download it via gdown.
    """
    device = torch.device(device)
    repo_ckpt = "face-parsing.PyTorch/79999_iter.pth"

    # 1) auto-download if you have gdown
    if not os.path.exists(repo_ckpt):
        if gdown is None:
            raise FileNotFoundError(
                f"{repo_ckpt} not found.\n"
                "Install gdown (`pip install gdown`) or download manually from:\n"
                "  https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812\n"
                f"and save it to {repo_ckpt}"
            )
        url = "https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
        os.makedirs(os.path.dirname(repo_ckpt), exist_ok=True)
        print(f"Downloading BiSeNet weights to {repo_ckpt}…")
        gdown.download(url, repo_ckpt, quiet=False)

    # 2) load into the network
    net = BiSeNet(n_classes=19).to(device)
    state = torch.load(repo_ckpt, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net

# ── 3.3  image → binary mask keeping labels 1–15 (face parts + hair) ───────
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def segment_face_hair(img_path, device="cpu"):
    """
    img_path : str
        Path to an RGB image.
    net      : BiSeNet model loaded by `load_parser`.
    Returns  : PIL.Image.Image  (same size as the input file)
    """
    # Load the BiSeNet model
    net = load_parser(device)
    
    # 1) Load original
    orig = Image.open(img_path).convert("RGB")
    w, h = orig.size

    # 2) Run at 512×512
    img_512 = orig.resize((512, 512), Image.BILINEAR)
    inp     = _transform(img_512).unsqueeze(0).to(device)
    with torch.no_grad():
        parsing = net(inp)[0].argmax(1)[0].cpu().numpy()  # H×W labels 0–18

    # 3) Define exactly the labels you want to keep:
    #    (background=0, skin=1, nose=2, eye_g=3, l_eye=4, r_eye=5,
    #     l_brow=6, r_brow=7, l_ear=8, r_ear=9, mouth=10, u_lip=11,
    #     l_lip=12, hair=13, ...)
    
    keep_ids = [
        1, 2,           # skin, nose
        3, 4, 5,        # eyes
        6, 7,           # brows
        8, 9,           # ears
        10, 11, 12,     # mouth + lips
        13,             # hat (if you want hats too; omit otherwise)
        17,             # hair (debug showed it here)
    ]

    # 4) Build the mask
    mask = np.isin(parsing, keep_ids).astype(np.uint8) * 255   # 0 or 255
    mask = Image.fromarray(mask).resize((w, h), Image.NEAREST)
    mask = np.array(mask) // 255                              # 0 or 1

    # 5) Apply it
    result_np = np.array(orig) * mask[:, :, None]             # broadcast RGB
    return Image.fromarray(result_np)

def segment_images_in_directory(input_dir, output_dir, device="cpu"):
    """
    Segments all images in a directory using `segment_face_hair` and saves the results to a new directory.

    input_dir : str
        Path to the directory containing input images.
    output_dir : str
        Path to the directory where segmented images will be saved.
    device : str
        Device to run the segmentation on ("cpu" or "cuda").
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # Segment the image
                segmented_image = segment_face_hair(input_path, device)

                # Save the segmented image to the output directory
                output_path = os.path.join(output_dir, filename)
                segmented_image.save(output_path)
                print(f"Segmented and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
        else:
            print(f"Skipping non-image file: {filename}")

def segment_images_from_csv(general_path, csv_path, device="cpu"):
    """
    Segments all images listed in a CSV using `segment_face_hair` and saves the results
    in the same location as the original images.

    csv_path : str
        Path to the CSV file containing image paths and their corresponding seasons.
    device : str
        Device to run the segmentation on ("cpu" or "cuda").
    """
    # Load the CSV
    df = pd.read_csv(os.path.join(general_path, csv_path))

    for idx, row in df.iterrows():
        image_path = os.path.join(general_path, row['image_path'])


        # Check if file exists and is an image
        if isinstance(image_path, str) and os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # Segment the image
                segmented_image = segment_face_hair(image_path, device)

                # Overwrite the original image
                segmented_image.save(image_path)
                print(f"Segmented and saved: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        else:
            print(f"Skipping invalid or non-image file: {image_path}")

def main():
    general_path = os.getcwd()
    path_csv = 'data/segmented_dataset/Paula/after/after_imgs.csv'
    segment_images_from_csv(general_path, path_csv)

if __name__ == "__main__":
    main()