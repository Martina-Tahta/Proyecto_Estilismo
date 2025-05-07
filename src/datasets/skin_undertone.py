import stone
import matplotlib.pyplot as plt
import cv2

def show_image_with_matplotlib(image, title=""):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


def skin_tone_detection(image_path, palette="perla", print_res=False):
    # map each perla hex (no “#”) → integer label 0–10
    perla_labels = {
        '373028': 0, '422811': 1, '513B2E': 2,
        '6F503C': 3, '81654F': 4, '9D7A54': 5,
        'BEA07E': 6, 'E5C8A6': 7, 'E7C1B8': 8,
        'F3DAD6': 9, 'FBF2F3': 10
    }

    # run the skin‑tone library
    result = stone.process(
        image_path,
        "color",
        palette,
        return_report_image=True
    )
    report_images = result.pop("report_images")
    if print_res:
        show_image_with_matplotlib(
            report_images[1],
            title="Skin Tone Classifier 1"
        )

    # pull out the two hex strings (strip ‘#’ and uppercase)
    face     = result['faces'][0]
    dom_hex  = face['dominant_colors'][0]['color'][1:].upper()
    skin_hex = face['skin_tone'][1:].upper()

    # normalize dominant‑color hex → [0.0,1.0]
    dom_int         = int(dom_hex, 16)
    normalized_dom  = dom_int / 0xFFFFFF

    # label‑encode skin tone
    label_skin = perla_labels.get(skin_hex)

    return [normalized_dom, label_skin]
