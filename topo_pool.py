import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttnPool(nn.Module):
    def __init__(self, input_dim, model_dim, num_latents=8, num_heads=8, ff_mult=2, dropout=0.0):
        super().__init__()
        assert model_dim % num_heads == 0
        H = num_heads; Hd = model_dim // num_heads
        self.latents = nn.Parameter(torch.randn(num_latents, model_dim) / math.sqrt(model_dim))

        self.ln_q = nn.LayerNorm(model_dim)
        self.ln_kv = nn.LayerNorm(input_dim)

        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        self.k_proj = nn.Linear(input_dim, model_dim, bias=False)
        nn.init.trunc_normal_(self.k_proj.weight, std=0.02)
        self.v_proj = nn.Linear(input_dim, model_dim, bias=False)
        nn.init.trunc_normal_(self.v_proj.weight, std=0.02)
        self.o_proj = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.trunc_normal_(self.o_proj.weight, std=0.02)

        self.ff = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, ff_mult * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * model_dim, model_dim),
        )

        nn.init.trunc_normal_(self.ff[1].weight, std=0.02)
        nn.init.zeros_(self.ff[1].bias)
        nn.init.trunc_normal_(self.ff[4].weight, std=0.02)
        nn.init.zeros_(self.ff[4].bias)
        self.num_heads, self.head_dim = H, Hd

    def forward(self, x_btsi: torch.Tensor) -> torch.Tensor:
        B, T, S, I = x_btsi.shape
        BT, H, Hd, L, M = B*T, self.num_heads, self.head_dim, self.latents.shape[0], self.latents.shape[1]
        x = x_btsi.reshape(BT, S, I)
        kv = self.ln_kv(x)

        q = self.ln_q(self.latents).unsqueeze(0).expand(BT, L, M)
        q = self.q_proj(q).view(BT, L, H, Hd).permute(0,2,1,3)   # [BT,H,L,Hd]
        k = self.k_proj(kv).view(BT, S, H, Hd).permute(0,2,1,3)  # [BT,H,S,Hd]
        v = self.v_proj(kv).view(BT, S, H, Hd).permute(0,2,1,3)  # [BT,H,S,Hd]

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)  # [BT,H,L,Hd]
        y = y.permute(0,2,1,3).contiguous().view(BT, L, M)
        y = self.o_proj(y)
        y = y + self.ff(y)
        return y.mean(dim=1).view(B, T, M)  # [B,T,model_dim]

class ProductEmbedTransform(nn.Module):
    def __init__(self, input_dim: int, model_dim: int,
                 num_directions: int = 128,         # D
                 # cross-attn
                 num_latents: int = 8, num_heads: int = 6, ff_mult: int = 2,
                 # low-rank hypernet
                 rank: int = 32,
                 # histogram CDF head
                 bins: int = 64,                    # K (try 32–64)
                 clip_sigma: float = 3.0,          # clip heights to ±clip_sigma after z-scoring
                 normalize_heights: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_directions = num_directions
        self.rank = rank
        self.bins = bins
        self.clip_sigma = clip_sigma
        self.normalize_heights = normalize_heights

        # 1) Cross-attention context over S
        self.ctx_pool = CrossAttnPool(input_dim, model_dim,
                                      num_latents=num_latents,
                                      num_heads=num_heads,
                                      ff_mult=ff_mult)

        # 2) Low-rank hypernet for directions: ctx -> U (I×rank), directions = U @ V
        self.phi = MLP(model_dim, model_dim, input_dim * rank)
        self.V = nn.Parameter(torch.randn(rank, num_directions) / math.sqrt(rank))

        self.cdf_weights = nn.Parameter(
            torch.randn(model_dim, bins, num_directions) / math.sqrt(bins * num_directions)
        )
        self.cdf_bias = nn.Parameter(torch.zeros(model_dim))

        # finish head
        self.proj_act = nn.SiLU()
        self.proj_fc2 = nn.Linear(model_dim, model_dim)
        nn.init.trunc_normal_(self.proj_fc2.weight, std=0.02)
        nn.init.zeros_(self.proj_fc2.bias)
        self.out_norm = nn.LayerNorm(model_dim)

    def _heights(self, x_btsi, ctx):
        """Compute heights without materializing directions: (x @ U) @ V."""
        B, T, S, I = x_btsi.shape
        U = self.phi(ctx).view(B, T, I, self.rank)           # [B,T,I,Rk]
        ht_r = torch.einsum('btsi,btir->btsr', x_btsi, U)    # [B,T,S,Rk]
        heights = torch.einsum('btsr,rd->btsd', ht_r, self.V)  # [B,T,S,D]
        return heights

    def _histogram_cdf(self, heights):
        """
        Differentiable histogram (linear binning) over S, then prefix-sum over bins.
        heights: [B,T,S,D]  ->  cdf: [B,T,bins,D]
        """
        B, T, S, D = heights.shape
        x = heights
        # z-score per [B,T,D] to keep things in-range, then clip
        if self.normalize_heights:
            mu = x.mean(dim=2, keepdim=True)
            sigma = x.std(dim=2, keepdim=True).clamp_min(1e-3)
            x = (x - mu) / sigma
        x = x.clamp(-self.clip_sigma, self.clip_sigma)
        # map to [0, K-1]
        K = self.bins
        u = ((x / self.clip_sigma) * 0.5 + 0.5) * (K - 1)    # [B,T,S,D] in [0,K-1]

        i0 = u.floor().clamp_(0, K - 1).to(torch.long)       # [B,T,S,D]
        i1 = (i0 + 1).clamp_(0, K - 1)                       # [B,T,S,D]
        w1 = (u - i0.to(u.dtype)).clamp_(0, 1)               # [B,T,S,D]
        w0 = 1.0 - w1

        BD = B * T * D
        i0f = i0.permute(0,1,3,2).reshape(BD, S)
        i1f = i1.permute(0,1,3,2).reshape(BD, S)
        w0f = w0.permute(0,1,3,2).reshape(BD, S)
        w1f = w1.permute(0,1,3,2).reshape(BD, S)

        hist = torch.zeros(BD, K, device=u.device, dtype=u.dtype)
        hist.scatter_add_(1, i0f, w0f)
        hist.scatter_add_(1, i1f, w1f)                       # [BD, K]

        cdf = hist.cumsum(dim=1) / S                         # prefix-sum over bins
        cdf = cdf.view(B, T, D, K).permute(0, 1, 3, 2)       # [B,T,K,D]
        return cdf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, E], with E % input_dim == 0 (E = S * input_dim)
        returns: [B, T, model_dim]
        """
        B, T, E = x.shape
        assert E % self.input_dim == 0, "last dim must be a multiple of input_dim"
        S = E // self.input_dim

        x_new = x.reshape(B, T, S, self.input_dim)           # [B,T,S,I]

        # 1) cross-attn context over S -> [B,T,M]
        ctx = self.ctx_pool(x_new)

        # 2) heights via low-rank contractions -> [B,T,S,D]
        heights = self._heights(x_new, ctx)

        # 3) histogram over S, then prefix-sum over bins -> [B,T,K,D]
        cdf = self._histogram_cdf(heights)

        # 4) contract (K,D) with weights -> [B,T,H], finish head
        y1 = torch.einsum('btkd,hkd->bth', cdf, self.cdf_weights) + self.cdf_bias
        y = self.proj_fc2(self.proj_act(y1))
        return self.out_norm(y)

