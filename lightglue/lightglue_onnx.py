import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts, size = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def pad_to_length(x, length) -> Tuple[torch.Tensor]:
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad = torch.ones(
        *x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
    )
    y = torch.cat([x, pad], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask

def swap_last_two(x) -> torch.Tensor:
    # swap the last two axes via transpose (onnx/torchscript friendly)
    return x.transpose(-1, -2).contiguous()


# def rotate_half(x) -> torch.Tensor:
#     # x = x.unflatten(-1, (-1, 2))
#     *prefix, T = x.shape
#     L = T // 2
#     x = x.reshape(*prefix, L, 2)  
#     x1, x2 = x.unbind(dim=-1)
#     return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

def rotate_half(x) -> torch.Tensor:
    B, H, M, T = x.size()       
    L = T // 2

    x = x.reshape(B, H, M, L, 2)

    x1 = x[..., 0]    # shape [B,H,M,L]
    x2 = x[..., 1]    # shape [B,H,M,L]

    y = torch.stack([-x2, x1], dim=-1)  # [B,H,M,L,2]

    return y.reshape(B, H, M, T)

def split_heads(t, heads : int):
    B, L, HD = t.shape
    H = heads
    D = HD // H
    t = t.reshape(B, L, H, D)
    return t.permute(0, 2, 1, 3).contiguous()

def transpose_and_flatten(t) -> torch.Tensor:
    # (B, H, M, D) -> (B, H, D*M)
    return t.transpose(1, 2).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs, t) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M, dim, F_dim = None, gamma = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        # emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        emb = torch.stack([cosines, sines], 0).unsqueeze(1)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0, desc1):
        """get confidence tokens"""
        # return (
        #     self.token(desc0.detach()).squeeze(-1),
        #     self.token(desc1.detach()).squeeze(-1),
        # )
        out0 = self.token(desc0.detach())  # [B, M, 1]
        out1 = self.token(desc1.detach())  # [B, N, 1]
        B0, M, _ = out0.shape
        B1, N, _ = out1.shape
        return out0.reshape(B0, M), out1.reshape(B1, N)



class Attention(nn.Module):
    def __init__(self, allow_flash) -> None:
        super().__init__()
        self.flash_available = FLASH_AVAILABLE
        if allow_flash and not self.flash_available:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and self.flash_available
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()
        if self.has_sdp:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask = None) -> torch.Tensor:
        
        if q.shape[-2] == 0 or k.shape[-2] == 0:
            # return q.new_zeros((*q.shape[:-1], v.shape[-1]))
            B, H, M, _ = q.size()
            Dv = v.size(-1)
            return q.new_zeros((B, H, M, Dv))
        # if self.enable_flash and q.device.type == "cuda":
        #     # use torch 2.0 scaled_dot_product_attention with flash
        #     if self.has_sdp:
        #         # args = [x.half().contiguous() for x in [q, k, v]]
        #         # v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
        #         # return v if mask is None else v.nan_to_num()
        #         qh = q.half().contiguous()
        #         kh = k.half().contiguous()
        #         vh = v.half().contiguous()
        #         v_out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask).to(q.dtype)
        #         if mask is None:
        #             return v_out
        #         else:
        #             return v_out.nan_to_num()

        #     else:
        #         assert mask is None
        #         # q, k, v = [x.transpose(-2, -3).contiguous() for x in [q, k, v]]
        #         # q, k, v = [swap_last_two(x) for x in (q, k, v)]
        #         q = q.permute(0, 2, 1, 3).contiguous()
        #         k = k.permute(0, 2, 1, 3).contiguous()
        #         v = v.permute(0, 2, 1, 3).contiguous()
        #         m = self.flash_(q.half(), torch.stack([k, v], 2).half())
        #         return m.permute(0, 2, 1, 3).to(q.dtype).clone()
        if self.has_sdp:
            # args = [x.contiguous() for x in [q, k, v]]
            # v = F.scaled_dot_product_attention(*args, attn_mask=mask)

            qc = q.contiguous()
            kc = k.contiguous()
            vc = v.contiguous()
            v = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            # attn = F.softmax(sim, -1)
            last_dim = sim.dim() - 1
            attn = F.softmax(sim, dim=last_dim)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim, num_heads, flash = False, bias = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x,
        encoding,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        # qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        B, M, total = qkv.shape
        H = self.num_heads
        D = total // (3 * H)
        qkv = qkv.reshape(-1, M, H, D, 3)  # [B, F, H, D, 3]
        qkv = qkv.permute(0, 2, 1, 3, 4).contiguous() # [B, H, F, D, 3]

        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2)) # [B, F, D]
        return x + self.ffn(torch.cat([x, message], 2))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim, num_heads, flash = False, bias = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.flash_available = FLASH_AVAILABLE
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and self.flash_available:
            self.flash = Attention(True)
        else:
            self.flash = None

    # def map_(self, func, x0, x1):
    #     return func(x0), func(x1)

    def forward(
        self, x0, x1
    ) -> List[torch.Tensor]:
        # qk0, qk1 = self.map_(self.to_qk, x0, x1)
        # v0, v1 = self.map_(self.to_v, x0, x1)
        qk0 = self.to_qk(x0)
        qk1 = self.to_qk(x1)
        v0 = self.to_v(x0)
        v1 = self.to_v(x1)
        # qk0, qk1, v0, v1 = map(
        #     lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
        #     (qk0, qk1, v0, v1),
        # )                
        qk0 = split_heads(qk0, self.heads)
        qk1 = split_heads(qk1, self.heads)
        v0  = split_heads(v0, self.heads)
        v1  = split_heads(v1, self.heads)

        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            # m1 = self.flash(
            #     qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            # )
            m1 = self.flash(
                qk1,
                qk0,
                v0,
                swap_last_two(mask) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            # attn01 = F.softmax(sim, dim=-1)
            # attn10 = F.softmax(swap_last_two(sim), dim=-1)
            last_dim = sim.dim() - 1
            attn01 = F.softmax(sim, dim=last_dim)
            sim_swapped = swap_last_two(sim)
            last_dim_swapped = sim_swapped.dim() - 1
            attn10 = F.softmax(sim_swapped, dim=last_dim_swapped)

            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum(
                "bhji, bhjd -> bhid",
                swap_last_two(attn10),
                v0
            )
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        # m0, m1 = self.map_(transpose_and_flatten, m0, m1)
        # m0, m1 = self.map_(self.to_out, m0, m1)
        m0 = transpose_and_flatten(m0)
        m1 = transpose_and_flatten(m1)
        m0 = self.to_out(m0)
        m1 = self.to_out(m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], 2))
        x1 = x1 + self.ffn(torch.cat([x1, m1], 2))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
    ):
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        return self.cross_attn(desc0, desc1)


def sigmoid_log_double_softmax(
    sim, z0, z1
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape    
    # certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    # scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores1 = swap_last_two(
        F.log_softmax(
            swap_last_two(sim).contiguous(),
            dim=2
        )
    )
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    # scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    # scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    ls0 = F.logsigmoid(-z0)         # [B, M, 1]
    B, M, _ = ls0.shape
    scores[:, :-1, -1] = ls0.reshape(B, M)   # [B, M]

    ls1 = F.logsigmoid(-z1)         # [B, N, 1]
    B, N, _ = ls1.shape
    scores[:, -1, :-1] = ls1.reshape(B, N)   # [B, N]

    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0, desc1):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc):
        out = torch.sigmoid(self.matchability(desc))
        B, M, _ = out.shape
        return out.reshape(B, M) 
        # return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores, th):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }

    # Point pruning involves an overhead (gather).
    # Therefore, we only activate it if there are enough keypoints.
    pruning_keypoint_thresholds = {
        "cpu": -1,
        "mps": -1,
        "cuda": 1024,
        "flash": 1536,
    }

    required_data_keys = ["image0", "image1"]

    version = "v0.1_arxiv"
    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    features = {
        "superpoint": {
            "weights": "superpoint_lightglue",
            "input_dim": 256,
        },
        "disk": {
            "weights": "disk_lightglue",
            "input_dim": 128,
        },
        "aliked": {
            "weights": "aliked_lightglue",
            "input_dim": 128,
        },
        "sift": {
            "weights": "sift_lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
        "doghardnet": {
            "weights": "doghardnet_lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
    }

    def __init__(self, features="superpoint", **conf) -> None:
        super().__init__()
        # self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        conf_ns = {**self.default_conf, **conf}
        # self.conf = conf = conf_ns

        self.depth_confidence  = conf_ns["depth_confidence"]
        self.width_confidence  = conf_ns["width_confidence"]
        self.filter_threshold  = conf_ns["filter_threshold"]
        self.n_layers          = conf_ns["n_layers"]
        self.num_heads         = conf_ns["num_heads"]
        self.input_dim         = conf_ns["input_dim"]
        self.descriptor_dim    = conf_ns["descriptor_dim"]
        self.flash             = conf_ns["flash"]
        self.mp                = conf_ns["mp"]
        self.add_scale_ori     = conf_ns["add_scale_ori"]
        self.weights           = conf_ns["weights"]        

        self.input_dim = 256
        self.weights = "superpoint_lightglue"

        self.flash_available = FLASH_AVAILABLE
        self.pruning_keypoint_thresholds = {
            "cpu": -1,
            "mps": -1,
            "cuda": 1024,
            "flash": 1536,
        }


        if self.input_dim != self.descriptor_dim:
            self.input_proj = nn.Linear(self.input_dim, self.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = self.descriptor_dim // self.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * self.add_scale_ori, head_dim, head_dim
        )

        h, n, d = self.num_heads, self.n_layers, self.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, self.flash) for _ in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor(
                [self.confidence_threshold(i) for i in range(self.n_layers)]
            ),
        )

        state_dict = None
        if features is not None:
            fname = f"{self.weights}_{self.version.replace('.', '-')}.pth"
            state_dict = torch.hub.load_state_dict_from_url(
                self.url.format(self.version, features), file_name=fname
            )
            self.load_state_dict(state_dict, strict=False)
        elif self.weights is not None:
            path = Path(__file__).parent
            path = path / "weights/{}.pth".format(self.weights)
            state_dict = torch.load(str(path), map_location="cpu")

        if state_dict:
            # rename old state dict entries
            for i in range(self.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = []

    def compile(
        self, mode="reduce-overhead", static_lengths=[256, 512, 768, 1024, 1280, 1536]
    ):
        if self.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        torch._inductor.cudagraph_mark_step_begin()
        for i in range(self.n_layers):
            self.transformers[i].masked_forward = torch.compile(
                self.transformers[i].masked_forward, mode=mode, fullgraph=True
            )

        self.static_lengths = static_lengths

    def forward(self, data) -> dict:
        """
        Match keypoints and descriptors between two images

        Input (dict):
            image0: dict
                keypoints: [B x M x 2]
                descriptors: [B x M x D]
                image: [B x C x H x W] or image_size: [B x 2]
            image1: dict
                keypoints: [B x N x 2]
                descriptors: [B x N x D]
                image: [B x C x H x W] or image_size: [B x 2]
        Output (dict):
            matches0: [B x M]
            matching_scores0: [B x M]
            matches1: [B x N]
            matching_scores1: [B x N]
            matches: List[[Si x 2]]
            scores: List[[Si]]
            stop
            prune0: [B x M]
            prune1: [B x N]
        """
        with torch.autocast(enabled=self.mp, device_type="cuda"):
            return self._forward(data)

    def _forward(self, data) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        size0, size1 = data0.get("image_size"), data1.get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        # if self.add_scale_ori:
        #     kpts0 = torch.cat(
        #         [kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")], -1
        #     )
        #     kpts1 = torch.cat(
        #         [kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")], -1
        #     )
        desc0 = data0["descriptors"].detach().contiguous()
        desc1 = data1["descriptors"].detach().contiguous()

        assert desc0.shape[-1] == self.input_dim
        assert desc1.shape[-1] == self.input_dim

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        mask0, mask1 = None, None
        c = max(m, n)
        do_compile = self.static_lengths and c <= max(self.static_lengths)

        if do_compile:
            kn = min([k for k in self.static_lengths if k >= c])
            desc0, mask0 = pad_to_length(desc0, kn)
            desc1, mask1 = pad_to_length(desc1, kn)
            kpts0, _ = pad_to_length(kpts0, kn)
            kpts1, _ = pad_to_length(kpts1, kn)
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.depth_confidence > 0
        do_point_pruning = self.width_confidence > 0 and not do_compile
        pruning_th = self.pruning_min_kpts(device)
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.n_layers):
            if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
                break
            desc0, desc1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1
            )
            if i == self.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                    break
            if do_point_pruning and desc0.shape[-2] > pruning_th:
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
            if do_point_pruning and desc1.shape[-2] > pruning_th:
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
            m0 = desc0.new_full((b, m), -1, dtype=torch.long)
            m1 = desc1.new_full((b, n), -1, dtype=torch.long)
            mscores0 = desc0.new_zeros((b, m))
            mscores1 = desc1.new_zeros((b, n))
            matches = desc0.new_empty((b, 0, 2), dtype=torch.long)
            mscores = desc0.new_empty((b, 0))
            if not do_point_pruning:
                prune0 = torch.ones_like(mscores0) * self.n_layers
                prune1 = torch.ones_like(mscores1) * self.n_layers
            return {
                "matches0": m0,
                "matches1": m1,
                "matching_scores0": mscores0,
                "matching_scores1": mscores1,
                "stop": i + 1,
                "matches": matches,
                "scores": mscores,
                "prune0": prune0,
                "prune1": prune1,
            }

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]  # remove padding
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.filter_threshold)
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        # TODO: Remove when hloc switches to the compact format.
        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.n_layers
            prune1 = torch.ones_like(mscores1) * self.n_layers
        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "stop": i + 1,
            "matches": matches,
            "scores": mscores,
            "prune0": prune0,
            "prune1": prune1,
        }

    def confidence_threshold(self, layer_index) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences, scores, layer_index
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            # keep |= confidences <= self.confidence_thresholds[layer_index]
            low_conf_mask = confidences <= self.confidence_thresholds[layer_index]
            keep = torch.logical_or(keep, low_conf_mask)
        return keep

    def check_if_stop(
        self,
        confidences0,
        confidences1,
        layer_index,
        num_points,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], 1)  
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.flash and self.flash_available and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]
