import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _MLPGenerator(nn.Module):
    def __init__(self, noise_dim, cond_dim, seq_len, feat_dim, hidden=256):
        super().__init__()
        in_dim = noise_dim + cond_dim
        out_dim = seq_len * feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),
        )
        self.seq_len = seq_len
        self.feat_dim = feat_dim

    def forward(self, z, c=None):
        if c is not None:
            z = torch.cat([z, c], dim=1)
        x = self.net(z)
        return x.view(z.size(0), self.seq_len, self.feat_dim)


class _MLPDiscriminator(nn.Module):
    def __init__(self, seq_len, feat_dim, cond_dim, hidden=256):
        super().__init__()
        in_dim = seq_len * feat_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, c=None):
        x = x.reshape(x.size(0), -1)
        if c is not None:
            x = torch.cat([x, c], dim=1)
        return self.net(x)


def trend_labels(seqs, target_dim=0):
    x0 = seqs[:, 0, target_dim]
    x1 = seqs[:, -1, target_dim]
    y = (x1 >= x0).astype(np.int64)
    return y


class WGANAugmentor:
    def __init__(self, seq_len, feat_dim, noise_dim=32, lr=1e-4, gp_lambda=10.0, n_critic=5, device='cpu'):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.noise_dim = noise_dim
        self.gp_lambda = gp_lambda
        self.n_critic = n_critic
        self.device = torch.device(device)
        self.G = _MLPGenerator(noise_dim, 0, seq_len, feat_dim).to(self.device)
        self.D = _MLPDiscriminator(seq_len, feat_dim, 0).to(self.device)
        self.opt_g = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
        self.opt_d = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.9))

    def _gradient_penalty(self, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, device=self.device)
        inter = alpha * real + (1 - alpha) * fake
        inter.requires_grad_(True)
        d_inter = self.D(inter)
        grad = torch.autograd.grad(
            outputs=d_inter,
            inputs=inter,
            grad_outputs=torch.ones_like(d_inter),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad = grad.view(grad.size(0), -1)
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def fit(self, real_np, steps=1000, batch_size=64):
        real = torch.tensor(real_np, dtype=torch.float32, device=self.device)
        n = real.size(0)
        for step in range(steps):
            for _ in range(self.n_critic):
                idx = torch.randint(0, n, (batch_size,), device=self.device)
                x_real = real[idx]
                z = torch.randn(batch_size, self.noise_dim, device=self.device)
                x_fake = self.G(z).detach()
                d_real = self.D(x_real).mean()
                d_fake = self.D(x_fake).mean()
                gp = self._gradient_penalty(x_real, x_fake)
                loss_d = d_fake - d_real + self.gp_lambda * gp
                self.opt_d.zero_grad()
                loss_d.backward()
                self.opt_d.step()

            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            x_fake = self.G(z)
            loss_g = -self.D(x_fake).mean()
            self.opt_g.zero_grad()
            loss_g.backward()
            self.opt_g.step()

            if step % max(1, steps // 5) == 0:
                print(f"[WGAN] step {step}/{steps} d={loss_d.item():.4f} g={loss_g.item():.4f}")

    def sample(self, n):
        self.G.eval()
        out = []
        bs = 256
        with torch.no_grad():
            for i in range(0, n, bs):
                cur = min(bs, n - i)
                z = torch.randn(cur, self.noise_dim, device=self.device)
                x = self.G(z).cpu().numpy()
                out.append(x)
        return np.concatenate(out, axis=0)


class CGANAugmentor:
    def __init__(self, seq_len, feat_dim, num_classes=2, noise_dim=32, lr=2e-4, device='cpu'):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.device = torch.device(device)
        self.G = _MLPGenerator(noise_dim, num_classes, seq_len, feat_dim).to(self.device)
        self.D = _MLPDiscriminator(seq_len, feat_dim, num_classes).to(self.device)
        self.opt_g = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.bce = nn.BCEWithLogitsLoss()

    def _one_hot(self, y):
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        return torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()

    def fit(self, real_np, labels_np, steps=1000, batch_size=64):
        real = torch.tensor(real_np, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels_np, dtype=torch.long, device=self.device)
        n = real.size(0)

        for step in range(steps):
            idx = torch.randint(0, n, (batch_size,), device=self.device)
            x_real = real[idx]
            y_real = labels[idx]
            c_real = self._one_hot(y_real)

            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            y_fake = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
            c_fake = self._one_hot(y_fake)
            x_fake = self.G(z, c_fake).detach()

            real_logit = self.D(x_real, c_real)
            fake_logit = self.D(x_fake, c_fake)
            d_loss = self.bce(real_logit, torch.ones_like(real_logit)) + self.bce(fake_logit, torch.zeros_like(fake_logit))

            self.opt_d.zero_grad()
            d_loss.backward()
            self.opt_d.step()

            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            y_gen = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
            c_gen = self._one_hot(y_gen)
            x_gen = self.G(z, c_gen)
            g_logit = self.D(x_gen, c_gen)
            g_loss = self.bce(g_logit, torch.ones_like(g_logit))

            self.opt_g.zero_grad()
            g_loss.backward()
            self.opt_g.step()

            if step % max(1, steps // 5) == 0:
                print(f"[CGAN] step {step}/{steps} d={d_loss.item():.4f} g={g_loss.item():.4f}")

    def sample(self, n, labels_np=None):
        self.G.eval()
        out = []
        bs = 256
        if labels_np is None:
            labels_np = np.random.randint(0, self.num_classes, size=n)
        labels_np = np.asarray(labels_np, dtype=np.int64)

        with torch.no_grad():
            for i in range(0, n, bs):
                cur = min(bs, n - i)
                y = labels_np[i : i + cur]
                c = self._one_hot(y)
                z = torch.randn(cur, self.noise_dim, device=self.device)
                x = self.G(z, c).cpu().numpy()
                out.append(x)
        return np.concatenate(out, axis=0)


def clone_opt(opt):
    return copy.deepcopy(opt)
