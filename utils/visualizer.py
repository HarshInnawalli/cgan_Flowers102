from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from config import sample_dir

def save_generated_images(generator, z, labels, epoch):
    with torch.no_grad():
        gen_imgs = generator(z, labels)
        gen_imgs = (gen_imgs + 1) / 2
        save_image(gen_imgs, f"{sample_dir}/epoch_{epoch}.png", nrow=10)

def show_final_image(epoch):
    img = mpimg.imread(f"{sample_dir}/epoch_{epoch}.png")
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Flower Images")
    plt.imshow(img)
    plt.show()
