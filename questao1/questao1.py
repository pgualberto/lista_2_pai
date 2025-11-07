import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATHS = {
    "Image 1": "pessoa.jpg",
    "Image 2": "pucminas.jpg",
    "Image 3": "vaticano.jpg"
}

def process_and_display(image_dict):

    num_images = len(image_dict)

    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))

    if num_images == 1:
        axes = np.array([axes])

    print("Starting processing...")

    for i, (title, path) in enumerate(image_dict.items()):

        img_original = cv2.imread(path)

        if img_original is None:
            print(f"Error: Could not load image '{path}'. Check the path.")
            for j in range(4):
                axes[i, j].text(0.5, 0.5, f"Error loading\n{path}",
                                horizontalalignment='center', verticalalignment='center',
                                color='red')
                axes[i, j].set_title(f"Image: {title}")
                axes[i, j].axis('off')
            continue

        img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        otsu_thresh, img_otsu = cv2.threshold(
            img_gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        canny_thresh_low = otsu_thresh * 0.5
        canny_thresh_high = otsu_thresh

        img_canny_auto = cv2.Canny(img_blur, canny_thresh_low, canny_thresh_high)

        axes[i, 0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{title} (Original)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_gray, cmap='gray')
        axes[i, 1].set_title("Grayscale")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img_otsu, cmap='gray')
        axes[i, 2].set_title(f"Otsu (Threshold: {otsu_thresh:.0f})")
        axes[i, 2].axis('off')

        axes[i, 3].imshow(img_canny_auto, cmap='gray')
        axes[i, 3].set_title(f"Canny (T_low={canny_thresh_low:.0f}, T_high={canny_thresh_high:.0f})")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.suptitle("Comparison: Otsu's Method vs. Canny Edge Detector", fontsize=20, y=1.03)

    output_filename = "comparative_visual_report.png"
    plt.savefig(output_filename)
    print(f"\nResults saved to '{output_filename}'")

    plt.show()

if __name__ == "__main__":
    process_and_display(IMAGE_PATHS)