"""sink_split: steer by letting the query choose between two halves of the attention sink (+ optional mean-diff residual).

Most heads put much of their attention on the first token (the attention sink). sink_split hides the real sink and puts two
copies in the KV cache, one carrying +v* and one −v*, with keys a small step apart along u. A query shift C·u moves
attention between the halves, so C sets how much of ±v* the heads read. v* is the value mean diff (sycophantic − abrasive)
at the last token. sink_split_resid adds mean_diff's residual vector at the same C.

Per attention layer L (full-attention layers ≥ 1), per KV head g, all after RoPE:
    sink±   key k_first ± ε·u,   value v_first ± ν·v̂*,   logit bias −ln 2 each   (k_first, v_first: this row's real first
            token, read from the cache at runtime)
    real first token hidden from query positions that see ≥ 4 real tokens (earlier ones keep it, so the sink itself is not rewritten)
    q_t += C·u                                   u: least-variance direction of real keys and queries, ⟂ mean sink key
    head write ≈ w_sink · ν · tanh(ε·(q_t·u + C)·scale) · v̂*
C = 0 is near, not exactly, bare: the halves split on q_t·u + C, and u is only approximately ⟂ the model's queries.
Measured on Qwen3.5-0.8B, ν = 320, 8 dev-cohort chat prompts: KL(bare || C=0) 0.005 nats averaged over positions, 0.014 at
the last token (a calibrated dose C0 is 1 nat); max |Δ log-prob| 3.85 (a rare token). Qwen3-4B, ν = 12.7: 0.015 nats over
positions, last token mean 0.18, max 1.43 (one of 8 prompts is a full dose off bare at its last token).
Qwen3.5-4B, ν = 550 (calibrated): 0.007 nats over positions, last token mean 0.015, max 0.043; max |Δ log-prob| 9.2.
TODO(PI[claude]): centre the split, e.g. subtract each head's mean q·u; untested.
sink_split_resid also:  h_L += C · r_scale · r̂*_L on mean_diff's default layers (20-80% depth)

Constants are set at extraction from iso-KL doses (1 nat RMS KL, steering-lite calibrate_iso_kl):
    ν_g = nu_mult · C0(sink value alone) · ‖v̂*_g‖        nu_mult 3.2 (Qwen3-4B: best −C dose 12.7 / C0 4.0)
    r_scale = C0(mean_diff) / C0(sink_split)                    so each part contributes at its own calibrated strength

Evidence (Qwen3-4B, BS-bench v2, Jev), write-up https://github.com/wassname/query-steering/blob/concept-steer/outputs/results.md
(sink_split_resid was named qslotr_sum there, sink_split q_slot_big): full 100 questions, −C side score sink_split_resid +3.94 vs mean_diff
+1.71, paired 90% CI of the difference [+1.79, +2.67]; 3 dev seeds agree; +C ties (Qwen3-4B already accepts most premises).
sink_split alone ≈ mean_diff. The random-vector control there was for a sibling method (sinkr_sum: a fixed v* written into the
sink value + residual): with a random unit vector in place of v* its −C gain fell to mean_diff's level. No random control
was run for sink_split / sink_split_resid themselves.
Requires attention that goes through transformers' ALL_ATTENTION_FUNCTIONS (Qwen3, Qwen3.5 full-attention layers).
PI[claude] 2026-09-29.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from einops import einsum
from loguru import logger

from ..config import SteeringConfig, register, register_config
from ..target import _get_blocks
from .vjp_resid import _encode, _unit_direction

MIN_VISIBLE = 4  # a query position reads the sink halves only once it sees this many real tokens (hand-set; 1-3 broke Qwen3-0.6B)
N_DIR_PROMPTS = 16  # prompts per class for u's key/query statistics (hand-set on Qwen3-4B)
CALIB_BRACKET = (1e-3, 4096.0)  # sink-write C0 was 4 on Qwen3-4B and 100 on Qwen3.5-0.8B
_ACTIVE: dict[int, dict] = {}  # id(attention module) -> slot state while attached


@contextmanager
def _record_qkv(model, layers):
    """record post-RoPE query/key/value per layer by wrapping the attention function -> {layer: (q, k, v)}"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    impl = _get_blocks(model)[layers[0]].self_attn.config._attn_implementation
    original = ALL_ATTENTION_FUNCTIONS[impl]
    wanted = {id(_get_blocks(model)[L].self_attn): L for L in layers}
    found = {}

    def record(module, query, key, value, attention_mask, **kw):
        if id(module) in wanted:
            found[wanted[id(module)]] = (query, key, value)
        return original(module, query, key, value, attention_mask, **kw)

    ALL_ATTENTION_FUNCTIONS[impl] = record
    try:
        yield found
    finally:
        ALL_ATTENTION_FUNCTIONS[impl] = original


@torch.no_grad()
def _value_mean_diff(model, tok, pos_prompts, neg_prompts, layers, batch_size, max_length):
    """v*_L = mean v⁺[last] − mean v⁻[last] -> {layer: [KVH, d]}"""
    sums = {}
    for sign, prompts in ((1.0, pos_prompts), (-1.0, neg_prompts)):
        for start in range(0, len(prompts), batch_size):
            batch = _encode(model, tok, prompts[start:start + batch_size], max_length)
            with _record_qkv(model, layers) as found:
                model(**batch)
            last = batch["attention_mask"].sum(1) - 1  # right padding
            rows = torch.arange(last.shape[0], device=last.device)
            for L in layers:
                v = found[L][2][rows, :, last].float()  # [b, KVH, d]
                sums[L] = sums.get(L, 0.0) + sign * v.sum(0) / len(prompts)
    return sums


@torch.no_grad()
def _directions(model, tok, prompts, layers, max_length):
    """u_g = least-variance direction of real keys + queries per layer, ⟂ the mean position-0 key (the sink; for chat prompts
    the same token starts every prompt, so this is the sink key the heads see)"""
    stats = {L: {"kk": 0.0, "qq": 0.0} for L in layers}
    sink = {}
    encoded = [tok(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length).input_ids for text in prompts]
    for ids in encoded:
        with _record_qkv(model, layers) as found:
            model(ids.to(next(model.parameters()).device))
        for L in layers:
            q, k, v = (x[0].float() for x in found[L])  # [heads, T, d]
            sink[L] = sink.get(L, 0.0) + k[:, 0].cpu() / len(encoded)
            kvh = k.shape[0]
            qg = q[:, 1:].reshape(kvh, -1, q.shape[-1])  # query heads of one KV group stacked as samples
            stats[L]["kk"] = stats[L]["kk"] + einsum(k[:, 1:], k[:, 1:], "g n d, g n e -> g d e").cpu()
            stats[L]["qq"] = stats[L]["qq"] + einsum(qg, qg, "g n d, g n e -> g d e").cpu()
    out = {}
    for L in layers:
        tr = lambda M: M.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
        M = stats[L]["kk"] / tr(stats[L]["kk"]) + stats[L]["qq"] / tr(stats[L]["qq"])
        k_sink = sink[L]
        u = torch.linalg.eigh(M).eigenvectors[..., 0]  # [KVH, d]
        u = u - (u * k_sink).sum(-1, keepdim=True) / (k_sink * k_sink).sum(-1, keepdim=True) * k_sink
        out[L] = u / u.norm(dim=-1, keepdim=True)
    return out


@torch.no_grad()
def _residual_mean_diff(model, tok, pos_prompts, neg_prompts, layers, batch_size, max_length):
    """r̂*_L = unit(mean h⁺[last] − mean h⁻[last]) at block L's output"""
    blocks, sums = _get_blocks(model), {}
    for sign, prompts in ((1.0, pos_prompts), (-1.0, neg_prompts)):
        for start in range(0, len(prompts), batch_size):
            batch = _encode(model, tok, prompts[start:start + batch_size], max_length)
            grabbed = {}
            hooks = [blocks[L].register_forward_hook(lambda _m, _i, o, L=L: grabbed.__setitem__(L, o[0] if isinstance(o, tuple) else o))
                     for L in layers]
            try:
                model(**batch)
            finally:
                for h in hooks:
                    h.remove()
            last = batch["attention_mask"].sum(1) - 1
            rows = torch.arange(last.shape[0], device=last.device)
            for L in layers:
                sums[L] = sums.get(L, 0.0) + sign * grabbed[L][rows, last].float().sum(0) / len(prompts)
    return {L: _unit_direction(s) for L, s in sums.items()}


def _default_resid_layers(n):  # walk.py resolve_layers default: 20-80% depth
    return tuple(range(max(2, int(n * 0.2)), min(n - 2, int(n * 0.8))))


def _iso_kl_c0(model, tok, cfg, shared, stacked, target_kl: float = 1.0) -> float:
    """iso-KL dose; raises if the solver did not reach the target (calibrate_iso_kl then returns a bracket endpoint)"""
    from ..calibrate import calibrate_iso_kl
    from ..vector import Vector
    c0, history = calibrate_iso_kl(Vector(cfg, shared, stacked), model, tok, None, target_kl=target_kl, target_stat="kl_rms",
                                   bracket=CALIB_BRACKET, device=next(model.parameters()).device, T=50, do_sample=True, seed=0)
    at = [h for h in history if math.isclose(h["coeff"], c0)]
    assert math.isfinite(c0) and c0 > 0 and at, f"{cfg.method}: iso-KL calibration gave c0={c0}"
    kl = at[-1]["kl_rms"]
    assert abs(kl - target_kl) <= 0.25 * target_kl, f"{cfg.method}: iso-KL c0={c0:.4g} reached kl_rms={kl:.3f}, target {target_kl}"
    return c0


def _extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
    from dataclasses import replace
    layers = tuple(cfg.layers)
    assert min(layers) >= 1, "layer 0: pos and neg end in the same token, so v* = 0 there"
    vstar = _value_mean_diff(model, tok, pos_prompts, neg_prompts, layers, batch_size, max_length)
    kv = _directions(model, tok, pos_prompts[:N_DIR_PROMPTS] + neg_prompts[:N_DIR_PROMPTS], layers, max_length)
    shared = {L: {"u": kv[L]} for L in layers}  # the sink k, v are read from the cache at runtime
    unit = {L: _unit_direction(vstar[L].flatten()).view_as(vstar[L]).cpu() for L in layers}  # ‖v̂*‖ = 1 over the layer
    # C0 of the sink write alone: both halves carry +v̂* write via ν = C: equivalent to sink_value (v_first += C·v̂*)
    c0_sink = None
    if cfg.nu_scale is None:  # given explicitly: use it (tests); else calibrate
        probe = replace(cfg, method="sink_split", layers=layers, nu_scale=1.0, sink_only=True, r_scale=0.0)
        c0_sink = _iso_kl_c0(model, tok, probe, {L: {**shared[L], "vstar": unit[L]} for L in layers}, {L: {"u": shared[L]["u"].flatten()[None]} for L in layers})
        cfg.nu_scale = cfg.nu_mult * c0_sink
    for L in layers:
        shared[L]["vstar"] = unit[L] * cfg.nu_scale
    stacked = {L: {"u": shared[L]["u"].flatten()[None]} for L in layers}
    if cfg.with_residual:
        r_layers = _default_resid_layers(len(_get_blocks(model)))
        r = _residual_mean_diff(model, tok, pos_prompts, neg_prompts, r_layers, batch_size, max_length)
        if cfg.r_scale is None:
            c0_attn = _iso_kl_c0(model, tok, replace(cfg, layers=layers, r_scale=0.0), shared, stacked)
            both = {**stacked, **{L: {**stacked.get(L, {}), "r": r[L][None].cpu()} for L in r_layers}}
            c0_resid = _iso_kl_c0(model, tok, replace(cfg, layers=tuple(sorted(both)), r_scale=1.0, attn_off=True), shared, both)
            cfg.r_scale = c0_resid / c0_attn
        for L in r_layers:
            stacked.setdefault(L, {})["r"] = r[L][None].cpu()
    cfg.layers = tuple(sorted(set(shared) | set(stacked)))  # attention layers ∪ residual layers: attach targets exactly these
    logger.info("sink_split C0 sink={} -> nu_scale={:.3g}; r_scale={}", c0_sink, cfg.nu_scale, cfg.r_scale if cfg.with_residual else None)
    return {L: {"shared": shared.get(L, {}), "stacked": stacked[L]} for L in sorted(set(shared) | set(stacked))}


def _slot_attention(original):
    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        P = _ACTIVE.get(id(module))
        if P is None:
            return original(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
        g = module.num_key_value_groups
        B, T, S = query.shape[0], query.shape[2], key.shape[2]
        if attention_mask is None:
            visible = torch.ones(T, S, dtype=torch.bool, device=query.device).tril(S - T)[None, None].expand(B, 1, T, S)
        elif attention_mask.dtype == torch.bool:
            visible = attention_mask[..., :S].expand(B, -1, T, S)
        else:  # additive float mask: masked entries are -inf or finfo(dtype).min
            visible = attention_mask[..., :S].expand(B, -1, T, S) > torch.finfo(attention_mask.dtype).min / 2
        seen = visible.sum(-1, keepdim=True)
        mature = seen >= MIN_VISIBLE
        first = visible.float().argmax(-1, keepdim=True)  # [B, 1, T, 1] first real key = the sink (left padding)
        rows = torch.arange(B, device=key.device)
        kf, vf = key[rows, :, first[:, 0, -1, 0]], value[rows, :, first[:, 0, -1, 0]]  # this row's sink k, v: [B, KVH, d]
        u, vs = (P[n].to(query) for n in ("u", "vstar"))
        if P["sink_only"]:  # calibration probe: every head reads its sink with +C·v̂* (= sink_value)
            slot_k, slot_v = torch.stack([kf, kf], 2), torch.stack([vf + P["C"] * vs, vf + P["C"] * vs], 2)
        else:
            slot_k = torch.stack([kf + P["eps"] * u, kf - P["eps"] * u], 2)  # [B, KVH, 2, d]
            slot_v = torch.stack([vf + vs, vf - vs], 2)
            query = query + P["C"] * u.repeat_interleave(g, 0)[None, :, None, :]
        k = torch.cat([key, slot_k.to(key)], 2).repeat_interleave(g, 1)
        v = torch.cat([value, slot_v.to(value)], 2).repeat_interleave(g, 1)
        logits = (query @ k.transpose(2, 3)).float() * scaling
        real = torch.zeros(visible.shape, device=query.device).masked_fill(~visible, -torch.inf)
        logits[..., :S] += real.masked_fill(torch.zeros_like(visible).scatter(-1, first, True) & mature, -torch.inf)
        logits[..., S:] = (logits[..., S:] - math.log(2)).masked_fill(~mature & (seen > 0), -torch.inf)  # seen == 0: padding rows
        A = logits.softmax(-1)
        return (A.to(v.dtype) @ v).transpose(1, 2).contiguous(), None
    return attention


class _Restore:
    def __init__(self, fn):
        self.remove = fn


def _install(model, cfg, stacked):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from ..attach import _gather_split_state
    blocks, hooks = _get_blocks(model), []
    attn_layers = [] if cfg.attn_off else [L for L, s in stacked.items() if "u" in s]
    if attn_layers:
        impl = blocks[attn_layers[0]].self_attn.config._attn_implementation
        original = ALL_ATTENTION_FUNCTIONS[impl]
        for L in attn_layers:
            sh = _gather_split_state(blocks[L])[0]
            attn = blocks[L].self_attn
            _ACTIVE[id(attn)] = {"vstar": sh["vstar"], "u": sh["u"], "eps": cfg.eps, "C": cfg.coeff, "sink_only": cfg.sink_only}
        ALL_ATTENTION_FUNCTIONS[impl] = _slot_attention(original)

        def restore():
            ALL_ATTENTION_FUNCTIONS[impl] = original
            for L in attn_layers:
                _ACTIVE.pop(id(blocks[L].self_attn), None)
        hooks.append(_Restore(restore))
    scale = cfg.r_scale or 0.0
    for L, s in stacked.items():
        if "r" in s and scale:
            def resid(_m, _i, out, r=s["r"].sum(0)):
                h = out[0] if isinstance(out, tuple) else out
                h = h + cfg.coeff * scale * r.to(h)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            hooks.append(blocks[L].register_forward_hook(resid))
    return hooks


@register_config
@dataclass
class SinkSplitC(SteeringConfig):
    method: str = "sink_split"
    eps: float = 1.0  # key offset along u; logit gap between the halves = 2·eps·(q·u + C)·scale (hand-set on Qwen3-0.6B: 0.05 too weak, 5 moved C=0)
    nu_mult: float = 3.2  # ν in units of the sink write's own C0 (hand-set: Qwen3-4B best −C dose 12.7 / C0 4.0)
    nu_scale: float | None = None  # set at extraction
    with_residual: bool = False
    r_scale: float | None = None  # set at extraction (sink_split_resid)
    sink_only: bool = False  # calibration probe only
    attn_off: bool = False  # calibration probe only


@register_config
@dataclass
class SinkSplitResidC(SinkSplitC):
    method: str = "sink_split_resid"
    with_residual: bool = True


for _name in ("sink_split", "sink_split_resid"):
    register(type(_name, (), {
        "name": _name, "extract_from_prompts": True, "cache_intervention": True,
        "extract": staticmethod(_extract), "install": staticmethod(_install),
        "apply": staticmethod(lambda _mod, _x, y, _shared, _stacked, _cfg: y),
    }))
