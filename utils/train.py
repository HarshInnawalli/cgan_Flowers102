import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from utils.visualizer import save_generated_images

def train(generator, discriminator, data_loader):
    bce_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real_imgs, labels) in enumerate(data_loader):
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            batch_size = real_imgs.size(0)
            valid = torch.full((batch_size,), 0.9, device=device)
            fake = torch.full((batch_size,), 0.1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, z_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)
            fake_pred, fake_feats = discriminator(gen_imgs, gen_labels)
            _, real_feats = discriminator(real_imgs, labels)
            fm_loss = nn.functional.l1_loss(fake_feats, real_feats.detach())
            g_loss = bce_loss(fake_pred, valid) + 10 * fm_loss
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_pred, _ = discriminator(real_imgs, labels)
            fake_pred, _ = discriminator(gen_imgs.detach(), gen_labels)
            d_loss = 0.5 * (bce_loss(real_pred, valid) + bce_loss(fake_pred, fake))
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(data_loader)}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

        z = torch.randn(num_classes, z_dim, device=device)
        sample_labels = torch.arange(num_classes, device=device)
        save_generated_images(generator, z, sample_labels, epoch+1)
