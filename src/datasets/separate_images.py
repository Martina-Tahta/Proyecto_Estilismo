import os
from PIL import Image
import csv


def separate_imgs_folder(folder, save_folder):
    season = os.path.basename(os.path.normpath(folder))

    folder_before = os.path.join(save_folder, os.path.join('before', season))
    folder_after = os.path.join(save_folder, os.path.join('after', season))
    os.makedirs(folder_before, exist_ok=True)
    os.makedirs(folder_after, exist_ok=True)

    ext_permitidas = ['.jpg', '.jpeg', '.png']

    for name in os.listdir(folder):
        if any(name.lower().endswith(ext) for ext in ext_permitidas):
            img_path = os.path.join(folder, name)
            img = Image.open(img_path)

            ancho, alto = img.size
            mitad = ancho // 2

            img_before = img.crop((0, 0, mitad, alto))
            img_after = img.crop((mitad, 0, ancho, alto))

            nombre_base = os.path.splitext(name)[0]
            ruta_antes = os.path.join(folder_before, f'{nombre_base}.jpg')
            ruta_despues = os.path.join(folder_after, f'{nombre_base}.jpg')

            img_before = img_before.convert('RGB')
            img_after = img_after.convert('RGB')

            img_before.save(ruta_antes)
            img_after.save(ruta_despues)

            print(f'Procesada: {name}')


def create_csv_with_season_labels(root_folder, output_csv, extensions=None):
    """
    Recursively walks through a folder and its subfolders,
    and creates a CSV with the relative path and the season (last folder in the path).

    Parameters:
    - root_folder: path to the main directory
    - output_csv: path to the CSV file to be created
    - extensions: list of allowed file extensions (e.g., ['.jpg', '.png']); if None, includes all files
    """
    data = []

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if extensions is None or any(filename.lower().endswith(ext) for ext in extensions):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, os.getcwd())
                season = os.path.basename(os.path.dirname(full_path))
                data.append([rel_path, season])

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'season'])  # Header
        writer.writerows(data)

    print(f"CSV created with {len(data)} entries at: {output_csv}")


def main():
    GENERAL_PATH = os.getcwd()
    # separate_imgs_folder(
    #     os.path.join(GENERAL_PATH,'data/raw/Paula/WINTER/true_winter'), 
    #     os.path.join(GENERAL_PATH, 'data/segmented_dataset/Paula'))
    
    path_imgs = os.path.join(GENERAL_PATH, 'data/segmented_dataset/Paula/after')
    create_csv_with_season_labels(path_imgs, os.path.join(path_imgs, 'after_imgs.csv'))

if __name__ == "__main__":
    main()