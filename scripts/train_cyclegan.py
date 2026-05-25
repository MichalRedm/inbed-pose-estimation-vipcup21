import os
import itertools
import argparse
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.data.dataset import VIPCupDataset
from src.models.cyclegan import GeneratorResNet, Discriminator, GANLoss

class CycleGANDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"] # This might be a tensor from DataAugmenter if used
        if isinstance(img, torch.Tensor) and img.max() <= 1.0:
            # Re-scale to -1, 1
            img = (img * 2.0) - 1.0
        else:
            img = self.transform(img)
        return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--max_steps", type=int, default=None, help="limit number of training steps per epoch for testing")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--data_dir", type=str, default="data/slp", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="where to save checkpoints")
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize networks
    # We use 3 channels because input_channels is typically 3 (RGB or Replicated IR)
    input_shape = (3, 256, 256)
    G_AB = GeneratorResNet(input_shape, num_residual_blocks=6)
    G_BA = GeneratorResNet(input_shape, num_residual_blocks=6)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)

    # Losses
    criterion_GAN = GANLoss().to(device)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Datasets
    print("Loading datasets...")
    # Domain A: Uncovered (Subjects 1-30)
    ds_A = VIPCupDataset(opt.data_dir, subjects=range(1, 31), covers=["uncover"], modalities=["IR"], split="train", in_channels=3)
    # Domain B: Covered (Subjects 31-80)
    ds_B = VIPCupDataset(opt.data_dir, subjects=range(31, 81), covers=["cover1", "cover2"], modalities=["IR"], split="train", in_channels=3)
    
    wrap_A = CycleGANDatasetWrapper(ds_A)
    wrap_B = CycleGANDatasetWrapper(ds_B)

    dataloader_A = DataLoader(wrap_A, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
    dataloader_B = DataLoader(wrap_B, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

    print(f"Domain A samples: {len(ds_A)}")
    print(f"Domain B samples: {len(ds_B)}")
    
    lambda_cyc = 10.0
    lambda_id = 5.0

    print("Starting training...")
    for epoch in range(opt.epochs):
        for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            if opt.max_steps is not None and i >= opt.max_steps:
                break
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # ------------------
            #  Train Generators
            # ------------------
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()

            # Identity loss
            # G_AB should be identity if real_B is fed
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), True)
            
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), True)
            
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            # Real loss
            loss_real = criterion_GAN(D_A(real_A), True)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(D_A(fake_A.detach()), False)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), True)
            loss_fake = criterion_GAN(D_B(fake_B.detach()), False)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{opt.epochs}] [Batch {i}/{len(dataloader_A)}] "
                      f"[D loss: {(loss_D_A + loss_D_B).item():.4f}] [G loss: {loss_G.item():.4f}, "
                      f"adv: {loss_GAN.item():.4f}, cycle: {loss_cycle.item():.4f}, id: {loss_identity.item():.4f}]")

        # Save model checkpoints
        if (epoch + 1) % 10 == 0 or (epoch + 1) == opt.epochs:
            torch.save(G_AB.state_dict(), os.path.join(opt.output_dir, "cyclegan_gen_A2B.pth"))
            torch.save(G_BA.state_dict(), os.path.join(opt.output_dir, "cyclegan_gen_B2A.pth"))
            torch.save(D_A.state_dict(), os.path.join(opt.output_dir, "cyclegan_disc_A.pth"))
            torch.save(D_B.state_dict(), os.path.join(opt.output_dir, "cyclegan_disc_B.pth"))
            print(f"Checkpoints saved to {opt.output_dir}")

if __name__ == "__main__":
    main()
