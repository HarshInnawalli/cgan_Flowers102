from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import get_data_loader
from utils.train import train
from utils.visualizer import show_final_image
from config import *

if __name__ == "__main__":
    data_loader = get_data_loader()
    generator = Generator(z_dim, num_classes).to(device)
    discriminator = Discriminator(num_classes).to(device)
    train(generator, discriminator, data_loader)
    show_final_image(epochs)
