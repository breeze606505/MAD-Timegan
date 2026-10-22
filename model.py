"""Network modules for TimeGAN and its multi-generator extension."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif "Conv" in classname:
        m.weight.data.normal_(0.0, 0.02)
    elif "Norm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)
    elif "GRU" in classname:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)


def _condition_dim(opt):
    use_condition = bool(getattr(opt, "use_class_conditioning", True))
    num_classes = int(getattr(opt, "num_classes", 0))
    return max(0, num_classes) if use_condition and num_classes > 1 else 0


def _expand_condition(condition, seq_len):
    if condition is None:
        return None
    if condition.dim() == 2:
        return condition.unsqueeze(1).expand(-1, int(seq_len), -1)
    return condition


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        data_dim = int(getattr(opt, "feature_dim", opt.z_dim))
        self.use_sigmoid = bool(getattr(opt, "use_sigmoid_output", True))
        self.rnn = nn.GRU(
            input_size=data_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer,
            batch_first=True,
        )
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=None):
        if sigmoid is None:
            sigmoid = self.use_sigmoid
        e_outputs, _ = self.rnn(input)
        h = self.fc(e_outputs)
        if sigmoid:
            h = self.sigmoid(h)
        return h


class Recovery(nn.Module):
    def __init__(self, opt):
        super(Recovery, self).__init__()
        data_dim = int(getattr(opt, "feature_dim", opt.z_dim))
        self.use_sigmoid = bool(getattr(opt, "use_sigmoid_output", True))
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=data_dim,
            num_layers=opt.num_layer,
            batch_first=True,
        )
        self.fc = nn.Linear(data_dim, data_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=None):
        if sigmoid is None:
            sigmoid = self.use_sigmoid
        r_outputs, _ = self.rnn(input)
        x_tilde = self.fc(r_outputs)
        if sigmoid:
            x_tilde = self.sigmoid(x_tilde)
        return x_tilde


class Generator(nn.Module):
    """Original GRU-based generator."""

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.cond_dim = _condition_dim(opt)
        self.use_sigmoid = bool(getattr(opt, "use_sigmoid_output", True))
        self.rnn = nn.GRU(
            input_size=opt.z_dim + self.cond_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer,
            batch_first=True,
        )
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, condition=None, sigmoid=None):
        if sigmoid is None:
            sigmoid = self.use_sigmoid
        cond_seq = _expand_condition(condition, input.size(1))
        if cond_seq is not None and self.cond_dim > 0:
            input = torch.cat([input, cond_seq], dim=-1)
        g_outputs, _ = self.rnn(input)
        e = self.fc(g_outputs)
        if sigmoid:
            e = self.sigmoid(e)
        return e


class TransGRUGenerator(nn.Module):
    """Transformer + GRU generator used to reproduce RESS-style improvement."""

    def __init__(self, opt):
        super(TransGRUGenerator, self).__init__()
        self.cond_dim = _condition_dim(opt)
        nhead = max(1, int(getattr(opt, "nhead", 4)))
        if opt.hidden_dim % nhead != 0:
            nhead = 1
        ff_dim = int(getattr(opt, "ffn_dim", opt.hidden_dim * 4))
        dropout = float(getattr(opt, "dropout", 0.1))
        num_tf_layers = int(getattr(opt, "transformer_layers", 1))
        self.in_proj = nn.Linear(opt.z_dim + self.cond_dim, opt.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=opt.hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_tf_layers)
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer,
            batch_first=True,
        )
        self.out_proj = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.use_residual = bool(getattr(opt, "gen_residual", True))
        self.use_layernorm = bool(getattr(opt, "gen_layernorm", True))
        self.pre_norm = nn.LayerNorm(opt.hidden_dim) if self.use_layernorm else nn.Identity()
        self.post_norm = nn.LayerNorm(opt.hidden_dim) if self.use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.use_sigmoid = bool(getattr(opt, "use_sigmoid_output", True))
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, condition=None, sigmoid=None):
        if sigmoid is None:
            sigmoid = self.use_sigmoid
        cond_seq = _expand_condition(condition, input.size(1))
        if cond_seq is not None and self.cond_dim > 0:
            input = torch.cat([input, cond_seq], dim=-1)
        x0 = self.in_proj(input)
        x = self.pre_norm(x0)
        x = self.transformer(x)
        if self.use_residual:
            x = x + x0
        x = self.dropout(x)
        g_outputs, _ = self.rnn(x)
        e = self.out_proj(g_outputs)
        if self.use_residual:
            e = e + g_outputs
        e = self.post_norm(e)
        if sigmoid:
            e = self.sigmoid(e)
        return e


class MultiGeneratorEnsemble(nn.Module):
    """Mixture-of-generators with a learned gating network."""

    def __init__(self, opt):
        super(MultiGeneratorEnsemble, self).__init__()
        self.opt = opt
        self.cond_dim = _condition_dim(opt)
        self.num_generators = max(1, int(getattr(opt, "num_generators", 1)))
        arch = getattr(opt, "generator_arch", "transgru").lower()
        gen_cls = TransGRUGenerator if arch == "transgru" else Generator
        self.generators = nn.ModuleList([gen_cls(opt) for _ in range(self.num_generators)])
        self.gate = nn.Linear(opt.z_dim * 2 + self.cond_dim, self.num_generators)
        self.apply(_weights_init)

    def forward(self, input, condition=None, return_details=False):
        outputs = [g(input, condition=condition) for g in self.generators]
        all_outputs = torch.stack(outputs, dim=0)
        gate_mean = input.mean(dim=1)
        gate_std = input.std(dim=1, unbiased=False)
        gate_in = torch.cat([gate_mean, gate_std], dim=-1)
        if condition is not None and self.cond_dim > 0:
            gate_in = torch.cat([gate_in, condition], dim=-1)
        weights = torch.softmax(self.gate(gate_in), dim=-1)
        mixed = torch.einsum("gbth,bg->bth", all_outputs, weights)
        if return_details:
            return mixed, weights, all_outputs
        return mixed

    def forward_single(self, input, g_idx, condition=None):
        g_idx = int(g_idx) % self.num_generators
        return self.generators[g_idx](input, condition=condition)

    @staticmethod
    def diversity_loss(all_outputs, margin=0.0):
        g_num = all_outputs.size(0)
        if g_num < 2:
            return all_outputs.new_zeros(())
        flat = all_outputs.reshape(g_num, all_outputs.size(1), -1)
        acc = 0.0
        pairs = 0
        for i in range(g_num):
            for j in range(i + 1, g_num):
                sim = F.cosine_similarity(flat[i], flat[j], dim=-1).mean()
                acc = acc + torch.relu(sim - margin)
                pairs += 1
        return acc / max(1, pairs)

    def prune_generators(self, keep_indices):
        keep_indices = sorted(set(int(i) for i in keep_indices if 0 <= int(i) < self.num_generators))
        if not keep_indices:
            keep_indices = [0]
        old_gate = self.gate
        self.generators = nn.ModuleList([self.generators[i] for i in keep_indices])
        self.num_generators = len(keep_indices)
        new_gate = nn.Linear(old_gate.in_features, self.num_generators).to(old_gate.weight.device)
        with torch.no_grad():
            new_gate.weight.copy_(old_gate.weight[keep_indices])
            new_gate.bias.copy_(old_gate.bias[keep_indices])
        self.gate = new_gate


class Supervisor(nn.Module):
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.use_sigmoid = bool(getattr(opt, "use_sigmoid_output", True))
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer,
            batch_first=True,
        )
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=None):
        if sigmoid is None:
            sigmoid = self.use_sigmoid
        s_outputs, _ = self.rnn(input)
        s = self.fc(s_outputs)
        if sigmoid:
            s = self.sigmoid(s)
        return s


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.cond_dim = _condition_dim(opt)
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(getattr(opt, "dropout", 0.0)))
        fc = nn.Linear(opt.hidden_dim + self.cond_dim, 1)
        if bool(getattr(opt, "disc_spectral_norm", False)):
            fc = nn.utils.spectral_norm(fc)
        self.fc = fc
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, condition=None, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        d_outputs = self.dropout(d_outputs)
        cond_seq = _expand_condition(condition, d_outputs.size(1))
        if cond_seq is not None and self.cond_dim > 0:
            d_outputs = torch.cat([d_outputs, cond_seq], dim=-1)
        y_hat = self.fc(d_outputs)
        if sigmoid:
            y_hat = self.sigmoid(y_hat)
        return y_hat
