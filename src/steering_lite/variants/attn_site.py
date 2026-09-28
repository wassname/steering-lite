"""Attention-site steering: the same contrast as mean_diff / vjp_delta, but read and written at a head's query, key or value.

Sites, per layer L (Qwen3 names, pre-RoPE, after q_norm / k_norm):
    q = q_norm(q_proj h)   [b s H d]      where a head looks
    k = k_norm(k_proj h)   [b s KVH d]    what is easy to find
    v = v_proj h           [b s KVH*d]    what a head reads out (goes into the KV cache)

Extraction (x = site activation):
    key_steer, value_steer:  x*_L = mean(x⁺_L[last]) − mean(x⁻_L[last])                   (query_steer does this for q)
    q_vjp, k_vjp:            c = mean h⁺_T[last] − mean h⁻_T[last],  T = n_layers − 3
                             g_L(x) = mean_{s∈V(x)} ∂(Σ_{t∈V} c·h_T,t)/∂x_L,s                (vjp_delta's estimator, source = site)
                             x*_L = mean⁺ g_L − mean⁻ g_L
Apply: x_L[t] += C · x̂*_L at every position; x̂* = x* / ‖x*‖ per layer.

Ref: wassname/vjp-steering (estimator), wassname/query-steering (q site). PI[claude] 2026-09-28.
"""
import math
from dataclasses import dataclass

import torch
from einops import einsum

from ..config import SteeringConfig, register, register_config
from ..target import _get_blocks
from .vjp_delta import _encode, _target_mean, _unit_direction, _valid_mask

ε = 1e-8
MODULE = {"q": "q_norm", "k": "k_norm", "v": "v_proj"}


def _site_modules(model, layers, site: str) -> dict[int, torch.nn.Module]:
    blocks = _get_blocks(model)
    return {layer: getattr(blocks[layer].self_attn, MODULE[site]) for layer in layers}


@torch.no_grad()
def _last_token_mean(model, tok, prompts, layers, site, batch_size, max_length) -> dict[int, torch.Tensor]:
    sums, grabbed = {layer: 0.0 for layer in layers}, {}
    hooks = [m.register_forward_hook(lambda _m, _i, out, layer=layer: grabbed.__setitem__(layer, out))
             for layer, m in _site_modules(model, layers, site).items()]
    try:
        for start in range(0, len(prompts), batch_size):
            batch = _encode(model, tok, prompts[start:start + batch_size], max_length)
            model(**batch)
            last = batch["attention_mask"].sum(1) - 1  # right padding
            rows = torch.arange(last.shape[0], device=last.device)
            for layer in layers:
                sums[layer] = sums[layer] + grabbed[layer][rows, last].float().sum(0)
    finally:
        for h in hooks:
            h.remove()
    return {layer: s / len(prompts) for layer, s in sums.items()}


def _site_gradients(model, tok, prompts, layers, site, target_layer, cotangent, skip_first, max_length):
    """per-prompt mean over valid positions of ∂(Σ_t∈V c·h_T,t)/∂x_L,s -> {layer: [b, F]}"""
    encoded = _encode(model, tok, prompts, max_length)
    valid = _valid_mask(encoded["attention_mask"], skip_first)
    assert valid.sum(1).min() > 0, f"a prompt has no valid positions after skip_first={skip_first}"
    found, root = {}, min(layers)

    def grab(layer):
        def hook(_m, _i, out):
            if layer == root:  # frozen model: seed the graph at the earliest site
                out = out.detach().requires_grad_(True)
            found[layer] = out
            return out
        return hook

    blocks = _get_blocks(model)
    hooks = [m.register_forward_hook(grab(layer)) for layer, m in _site_modules(model, layers, site).items()]
    hooks.append(blocks[target_layer].register_forward_hook(lambda _m, _i, out: found.__setitem__("T", out[0] if isinstance(out, tuple) else out)))
    try:
        with torch.enable_grad():
            model(**encoded)
            target = found["T"]
            grads = torch.autograd.grad(target, [found[layer] for layer in layers],
                                        grad_outputs=cotangent.to(target)[None, None, :] * valid[..., None])
    finally:
        for h in hooks:
            h.remove()
    counts = valid.sum(1).float()
    out = {}
    for layer, g in zip(layers, grads, strict=True):
        g = g.float().flatten(2)  # [b s F]
        out[layer] = einsum(g, valid.float(), "b s f, b s -> b f") / counts[:, None]
    return out


def _class_mean_site_vjp(model, tok, prompts, layers, site, target, cotangent, batch_size, max_length, skip_first):
    totals = {}
    for start in range(0, len(prompts), batch_size):
        grads = _site_gradients(model, tok, prompts[start:start + batch_size], layers, site, target, cotangent, skip_first, max_length)
        for layer, g in grads.items():
            totals[layer] = totals.get(layer, 0.0) + g.sum(0)
    return {layer: t / len(prompts) for layer, t in totals.items()}


def _install_add(model, cfg, stacked, site):
    """x ← x + C·x̂ at every position; x̂ is stored flat [1, F] and reshaped to the module output's trailing dims."""
    def hook(_m, _i, out, x):
        return out + cfg.coeff * x.to(out).view(out.shape[2:])
    return [m.register_forward_hook(lambda _m, _i, out, x=stacked[layer]["x"].sum(0): hook(_m, _i, out, x))
            for layer, m in _site_modules(model, stacked, site).items()]


def _pack(directions):
    return {layer: {"shared": {}, "stacked": {"x": d.flatten().unsqueeze(0)}} for layer, d in directions.items()}


def _layers(model, cfg):
    return tuple(range(len(_get_blocks(model)))) if cfg.layers is None else tuple(cfg.layers)


def _mean_diff_site(site):
    def extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
        layers = _layers(model, cfg)
        pos = _last_token_mean(model, tok, pos_prompts, layers, site, batch_size, max_length)
        neg = _last_token_mean(model, tok, neg_prompts, layers, site, batch_size, max_length)
        return _pack({layer: _unit_direction(pos[layer] - neg[layer]) for layer in layers})
    return extract


def _vjp_site(site):
    def extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
        model.requires_grad_(False)
        count = len(_get_blocks(model))
        target = count - 3 if cfg.target_layer is None else cfg.target_layer
        layers = tuple(layer for layer in _layers(model, cfg) if layer < target)
        c = _target_mean(model, tok, pos_prompts, target, batch_size, max_length) - _target_mean(model, tok, neg_prompts, target, batch_size, max_length)
        pos = _class_mean_site_vjp(model, tok, pos_prompts, layers, site, target, c, batch_size, max_length, cfg.skip_first)
        neg = _class_mean_site_vjp(model, tok, neg_prompts, layers, site, target, c, batch_size, max_length, cfg.skip_first)
        return _pack({layer: _unit_direction(pos[layer] - neg[layer]) for layer in layers})
    return extract


def _method(name, site, extract):
    cls = type(name, (), {
        "name": name, "extract_from_prompts": True, "cache_intervention": True,  # own hooks via install()
        "extract": staticmethod(extract),
        "install": staticmethod(lambda model, cfg, stacked: _install_add(model, cfg, stacked, site)),
        "apply": staticmethod(lambda _mod, _x, y, _shared, _stacked, _cfg: y),
    })
    return register(cls)


@register_config
@dataclass
class KeySteerC(SteeringConfig):
    method: str = "key_steer"


@register_config
@dataclass
class ValueSteerC(SteeringConfig):
    method: str = "value_steer"


@register_config
@dataclass
class QVjpC(SteeringConfig):
    method: str = "q_vjp"
    target_layer: int | None = None
    skip_first: int = 16


@register_config
@dataclass
class KVjpC(SteeringConfig):
    method: str = "k_vjp"
    target_layer: int | None = None
    skip_first: int = 16


KeySteer = _method("key_steer", "k", _mean_diff_site("k"))
ValueSteer = _method("value_steer", "v", _mean_diff_site("v"))
QVjp = _method("q_vjp", "q", _vjp_site("q"))
KVjp = _method("k_vjp", "k", _vjp_site("k"))


# --- the two pathways together -------------------------------------------------------------------------------------
# r*_L = mean(h⁺_L[last]) − mean(h⁻_L[last]) at block L's output (what mean_diff adds).
#
# q_retrieve ("the Q that retrieves r*"): per layer, the query shift whose attention output writes most along r*_L,
#     q*_L = mean_{x∈pos∪neg} ∇_δ Σ_{t∈V(x)} ⟨r̂*_L, o_proj_L(attn(q + δ, k, v))_t⟩
#     local to the layer: each block's input is detached, so δ_L reaches only its own o_proj.
# qr_sum: mean_diff on the residual (20-80% depth) and query_steer on the attention layers, one coefficient:
#     h_L ← h_L + C·r_scale·r̂*_L,   q_L ← q_L + C·q̂*_L        (r_scale ≈ C0_mean_diff / C0_query_steer on Qwen3-4B)


@torch.no_grad()
def _last_token_residual(model, tok, prompts, layers, batch_size, max_length) -> dict[int, torch.Tensor]:
    blocks, sums, grabbed = _get_blocks(model), {layer: 0.0 for layer in layers}, {}
    hooks = [blocks[layer].register_forward_hook(lambda _m, _i, out, layer=layer: grabbed.__setitem__(layer, out[0] if isinstance(out, tuple) else out))
             for layer in layers]
    try:
        for start in range(0, len(prompts), batch_size):
            batch = _encode(model, tok, prompts[start:start + batch_size], max_length)
            model(**batch)
            last = batch["attention_mask"].sum(1) - 1
            rows = torch.arange(last.shape[0], device=last.device)
            for layer in layers:
                sums[layer] = sums[layer] + grabbed[layer][rows, last].float().sum(0)
    finally:
        for h in hooks:
            h.remove()
    return {layer: s / len(prompts) for layer, s in sums.items()}


def _residual_star(model, tok, pos_prompts, neg_prompts, layers, batch_size, max_length):
    pos = _last_token_residual(model, tok, pos_prompts, layers, batch_size, max_length)
    neg = _last_token_residual(model, tok, neg_prompts, layers, batch_size, max_length)
    return {layer: _unit_direction(pos[layer] - neg[layer]) for layer in layers}


def _retrieve_gradients(model, tok, prompts, layers, r_hat, skip_first, max_length):
    encoded = _encode(model, tok, prompts, max_length)
    valid = _valid_mask(encoded["attention_mask"], skip_first)
    assert valid.sum(1).min() > 0
    blocks, found, hooks = _get_blocks(model), {}, []
    B, H, d = len(prompts), blocks[layers[0]].self_attn.config.num_attention_heads, blocks[layers[0]].self_attn.head_dim
    δ = {layer: torch.zeros(B, 1, H, d, device=valid.device, requires_grad=True) for layer in layers}
    for i, block in enumerate(blocks):  # cut the residual path between layers, so each δ_L reaches only o_proj_L
        hooks.append(block.register_forward_pre_hook(lambda _m, args, kwargs: ((args[0].detach(), *args[1:]), kwargs), with_kwargs=True))
    for layer in layers:
        attn = blocks[layer].self_attn
        hooks.append(attn.q_norm.register_forward_hook(lambda _m, _i, out, layer=layer: out + δ[layer].to(out)))
        hooks.append(attn.o_proj.register_forward_hook(lambda _m, _i, out, layer=layer: found.__setitem__(layer, out)))
    try:
        with torch.enable_grad():
            model(**encoded)
            objective = sum(einsum(found[layer].float(), r_hat[layer].to(found[layer].device), valid.float(), "b s D, D, b s -> ")
                            for layer in layers)
            grads = torch.autograd.grad(objective, [δ[layer] for layer in layers])
    finally:
        for h in hooks:
            h.remove()
    return {layer: g[:, 0].sum(0) / valid.sum(1).float().mean() for layer, g in zip(layers, grads, strict=True)}  # [H, d]


def _retrieve_sum(model, tok, prompts, layers, r_hat, cfg, batch_size, max_length):
    totals = {}
    for start in range(0, len(prompts), batch_size):
        for layer, g in _retrieve_gradients(model, tok, prompts[start:start + batch_size], layers, r_hat, cfg.skip_first, max_length).items():
            totals[layer] = totals.get(layer, 0.0) + g
    return totals


def _q_retrieve_extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
    """estimator "mean": sum of g over pos ∪ neg;  "delta": Σ_pos g − Σ_neg g (vjp_delta style)"""
    model.requires_grad_(False)
    layers = _layers(model, cfg)
    r_hat = _residual_star(model, tok, pos_prompts, neg_prompts, layers, batch_size, max_length)
    gp = _retrieve_sum(model, tok, pos_prompts, layers, r_hat, cfg, batch_size, max_length)
    gn = _retrieve_sum(model, tok, neg_prompts, layers, r_hat, cfg, batch_size, max_length)
    sign = {"mean": 1.0, "delta": -1.0}[cfg.estimator]
    return _pack({layer: _unit_direction(gp[layer] + sign * gn[layer]) for layer in layers})


def _default_resid_layers(n):  # walk.py resolve_layers default: 20-80% depth
    return tuple(range(max(2, int(n * 0.2)), min(n - 2, int(n * 0.8))))


def _qr_sum_extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
    layers = _layers(model, cfg)
    r_layers = _default_resid_layers(len(_get_blocks(model)))
    q_extract = {"dom": _mean_diff_site("q"), "retrieve": _q_retrieve_extract}[cfg.q_source]
    q = q_extract(model, tok, pos_prompts, neg_prompts, cfg, batch_size=batch_size, max_length=max_length)
    r = _residual_star(model, tok, pos_prompts, neg_prompts, r_layers, batch_size, max_length)
    out = {layer: {"shared": {}, "stacked": {}} for layer in sorted(set(layers) | set(r_layers))}
    for layer in layers:
        out[layer]["stacked"]["x"] = q[layer]["stacked"]["x"]
    for layer in r_layers:
        out[layer]["stacked"]["r"] = r[layer].unsqueeze(0)
    return out


def _qr_sum_install(model, cfg, stacked):
    blocks, hooks = _get_blocks(model), []
    for layer, s in stacked.items():
        if "x" in s:
            hooks += _install_add(model, cfg, {layer: {"x": s["x"]}}, "q")
        if "r" in s:
            def resid(_m, _i, out, r=s["r"].sum(0)):
                h = out[0] if isinstance(out, tuple) else out
                h = h + cfg.coeff * cfg.r_scale * r.to(h)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            hooks.append(blocks[layer].register_forward_hook(resid))
    return hooks


@register_config
@dataclass
class QRetrieveC(SteeringConfig):
    method: str = "q_retrieve"
    skip_first: int = 16
    estimator: str = "mean"


@register_config
@dataclass
class QRetrieveDeltaC(QRetrieveC):
    method: str = "q_retrieve_delta"
    estimator: str = "delta"


@register_config
@dataclass
class QRSumC(SteeringConfig):
    method: str = "qr_sum"
    r_scale: float = 0.1
    q_source: str = "dom"
    skip_first: int = 16
    estimator: str = "mean"


@register_config
@dataclass
class QRetrSumC(QRSumC):
    method: str = "qretr_sum"
    r_scale: float = 1.5  # C0 mean_diff 3.01 / C0 q_retrieve 1.96 on Qwen3-4B dev
    q_source: str = "retrieve"


QRetrieve = _method("q_retrieve", "q", _q_retrieve_extract)
QRetrieveDelta = _method("q_retrieve_delta", "q", _q_retrieve_extract)
for _name in ("qr_sum", "qretr_sum"):
    register(type(_name, (), {
        "name": _name, "extract_from_prompts": True, "cache_intervention": True,
        "extract": staticmethod(_qr_sum_extract), "install": staticmethod(_qr_sum_install),
        "apply": staticmethod(lambda _mod, _x, y, _shared, _stacked, _cfg: y),
    }))


# --- attention-gated write through the attention sink -----------------------------------------------------------
# Most heads put much of their attention on the first token (the sink). Edit only that token's value:
#     v_L[first] ← v_L[first] + C·v̂*_L        (prefill only; the edited value stays in the KV cache for every later token)
# so head h writes W_O^h (A_t,first · C v̂*) into the residual at each position t: a steering write gated by attention.
#     sink_write:  v*_L,g = Σ_{h∈g} W_O^hᵀ r̂*_L    (the value that makes the group's heads write r*, mean_diff's direction)
#     sink_value:  v*_L = mean(v⁺[last]) − mean(v⁻[last])   (value_steer's direction, sink only)


def _wo_transpose_r(model, layers, r_hat):
    blocks, out = _get_blocks(model), {}
    for layer in layers:
        attn = blocks[layer].self_attn
        H, KVH, d = attn.config.num_attention_heads, attn.config.num_key_value_heads, attn.head_dim
        w = (attn.o_proj.weight.float().T @ r_hat[layer].to(attn.o_proj.weight.device)).view(KVH, H // KVH, d).sum(1)  # [KVH, d]
        out[layer] = _unit_direction(w.flatten())
    return out


def _sink_write_extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
    layers = _layers(model, cfg)
    r_hat = _residual_star(model, tok, pos_prompts, neg_prompts, layers, batch_size, max_length)
    return _pack(_wo_transpose_r(model, layers, r_hat))


def _punct_ids(tok_vocab_size, model):
    """token ids made only of punctuation / whitespace with at least one of . , ; : ! ? newline (full stops etc. are secondary sinks)"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model.config._name_or_path)
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    return torch.tensor([i for i, t in enumerate(vocab) if any(c in t for c in ".\n,;:!?Ċ") and len(t.strip("Ġ ĊĉĠ.,;:!?\n")) == 0])


def _install_sink(model, cfg, stacked):
    """add C·x̂ to the value of the first real token (prefill); with cfg.punct also to every punctuation token, including decoded ones"""
    mask = {}
    punct = _punct_ids(None, model).to(next(model.parameters()).device) if getattr(cfg, "punct", False) else None

    def grab_mask(_m, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        am = kwargs.get("attention_mask")
        mask["first"] = None if am is None or ids is None or am.shape[1] != ids.shape[1] else am.long().argmax(1)  # left padding
        mask["punct"] = None if punct is None or ids is None else torch.isin(ids, punct)
        return None

    def hook(_m, _i, out, x):
        sel = torch.zeros(out.shape[:2], dtype=torch.bool, device=out.device)
        if out.shape[1] > 1:  # prefill: the first real token; decode steps find it already in the cache
            rows = torch.arange(out.shape[0], device=out.device)
            sel[rows, mask["first"] if mask["first"] is not None else torch.zeros_like(rows)] = True
        if mask.get("punct") is not None and mask["punct"].shape == sel.shape:
            sel |= mask["punct"]
        return out + cfg.coeff * sel[..., None].to(out) * x.to(out)

    hooks = [model.register_forward_pre_hook(grab_mask, with_kwargs=True)]
    hooks += [m.register_forward_hook(lambda _m, _i, out, x=stacked[layer]["x"].sum(0): hook(_m, _i, out, x))
              for layer, m in _site_modules(model, stacked, "v").items()]
    return hooks


@register_config
@dataclass
class SinkWriteC(SteeringConfig):
    method: str = "sink_write"


@register_config
@dataclass
class SinkValueC(SteeringConfig):
    method: str = "sink_value"


@register_config
@dataclass
class SinkPunctC(SteeringConfig):
    method: str = "sink_punct"
    punct: bool = True


for _name, _extract in (("sink_write", _sink_write_extract), ("sink_value", _mean_diff_site("v")), ("sink_punct", _mean_diff_site("v"))):
    register(type(_name, (), {
        "name": _name, "extract_from_prompts": True, "cache_intervention": True,
        "extract": staticmethod(_extract), "install": staticmethod(_install_sink),
        "apply": staticmethod(lambda _mod, _x, y, _shared, _stacked, _cfg: y),
    }))


# sinkr_sum: sink_value (attention sink's value) + mean_diff (residual, 20-80% depth), one coefficient:
#     v_L[first] += C·v̂*_L,   h_L += C·r_scale·r̂*_L      (r_scale = C0 mean_diff / C0 sink_value ≈ 3.0/4.0 on Qwen3-4B dev)
def _sinkr_extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
    v = _mean_diff_site("v")(model, tok, pos_prompts, neg_prompts, cfg, batch_size=batch_size, max_length=max_length)
    r_layers = _default_resid_layers(len(_get_blocks(model)))
    r = _residual_star(model, tok, pos_prompts, neg_prompts, r_layers, batch_size, max_length)
    out = {layer: {"shared": {}, "stacked": {}} for layer in sorted(set(v) | set(r_layers))}
    for layer in v:
        out[layer]["stacked"]["x"] = v[layer]["stacked"]["x"]
    for layer in r_layers:
        out[layer]["stacked"]["r"] = r[layer].unsqueeze(0)
    return out


def _sinkr_install(model, cfg, stacked):
    blocks = _get_blocks(model)
    hooks = _install_sink(model, cfg, {layer: {"x": s["x"]} for layer, s in stacked.items() if "x" in s})
    for layer, s in stacked.items():
        if "r" in s:
            def resid(_m, _i, out, r=s["r"].sum(0)):
                h = out[0] if isinstance(out, tuple) else out
                h = h + cfg.coeff * cfg.r_scale * r.to(h)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            hooks.append(blocks[layer].register_forward_hook(resid))
    return hooks


@register_config
@dataclass
class SinkRSumC(SteeringConfig):
    method: str = "sinkr_sum"
    r_scale: float = 0.75


register(type("sinkr_sum", (), {
    "name": "sinkr_sum", "extract_from_prompts": True, "cache_intervention": True,
    "extract": staticmethod(_sinkr_extract), "install": staticmethod(_sinkr_install),
    "apply": staticmethod(lambda _mod, _x, y, _shared, _stacked, _cfg: y),
}))


# --- q_prefix: the working q-steer, plus something for it to find --------------------------------------------------
# Query steering fetched a password because the password was in the context. A persona is not in the test prompt, so
# put the persona sentences in the cache, hidden by a logit bias, and let the same q* decide how much is read:
#     q*_L = mean(q⁺_L[last]) − mean(q⁻_L[last])                 (query_steer's extraction, unchanged)
#     prefix± = K, V of "Answer as someone who is {sycophantic | abrasive}." (post-RoPE at their own positions 1..n)
#     q_t += C·q̂*_L;   logit(t, s ∈ prefix) = q_t·k_s·scale − b        (b: prefix nearly unread at C = 0)
PERSONA_SENTENCES = ("Answer as someone who is sycophantic.", "Answer as someone who is abrasive.")
MIN_VISIBLE = 4  # a query position reads the prefix only once it can see this many real tokens
_ACTIVE_PREFIX: dict[int, dict] = {}  # id(attention module) -> {"k": [KVH, n, d], "v": [KVH, n, d], "b": float}


@torch.no_grad()
def _persona_kv(model, tok, layers):
    """K (post-RoPE), V of each persona sentence inside a user turn, first (sink) token dropped -> {layer: (k, v)}"""
    from transformers import AutoTokenizer
    tok = tok or AutoTokenizer.from_pretrained(model.config._name_or_path)
    blocks, out = _get_blocks(model), {layer: ([], []) for layer in layers}
    for sentence in PERSONA_SENTENCES:
        ids = tok(tok.apply_chat_template([{"role": "user", "content": sentence}], tokenize=False).split(sentence)[0] + sentence,
                  return_tensors="pt", add_special_tokens=False).input_ids.to(next(model.parameters()).device)
        grabbed = {}
        hooks = [blocks[layer].self_attn.k_norm.register_forward_hook(lambda _m, _i, o, layer=layer: grabbed.__setitem__(("k", layer), o)) for layer in layers]
        hooks += [blocks[layer].self_attn.v_proj.register_forward_hook(lambda _m, _i, o, layer=layer: grabbed.__setitem__(("v", layer), o)) for layer in layers]
        try:
            model(ids)
        finally:
            for h in hooks:
                h.remove()
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        cos, sin = model.model.rotary_emb(grabbed[("k", layers[0])].transpose(1, 2), pos)
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        for layer in layers:
            attn = blocks[layer].self_attn
            k = grabbed[("k", layer)].transpose(1, 2)  # [1, KVH, n, d]
            k, _ = apply_rotary_pos_emb(k, k, cos, sin)
            v = grabbed[("v", layer)].view(1, ids.shape[1], -1, attn.head_dim).transpose(1, 2)
            out[layer][0].append(k[0, :, 1:].float())  # drop the first token (the sink)
            out[layer][1].append(v[0, :, 1:].float())
    return {layer: (torch.cat(ks, 1), torch.cat(vs, 1)) for layer, (ks, vs) in out.items()}, [int(t.shape[1]) for t in out[layers[0]][0]]


def _q_prefix_extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
    q = _mean_diff_site("q")(model, tok, pos_prompts, neg_prompts, cfg, batch_size=batch_size, max_length=max_length)
    kv, lengths = _persona_kv(model, tok, tuple(q))
    for layer in q:
        q[layer]["shared"] = {"pk": kv[layer][0], "pv": kv[layer][1], "npos": torch.tensor([lengths[0]])}
    return q


def _prefix_attention(original):
    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        P = _ACTIVE_PREFIX.get(id(module))
        if P is None:
            return original(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
        g = module.num_key_value_groups
        if "dq" in P:  # q_prefix_k: q_t (post-RoPE) += C·(k⁺ − k⁻)/‖·‖, per KV head broadcast to its query heads
            query = query + P["C"] * P["dq"].to(query).repeat_interleave(g, 0)[None, :, None, :]
        k = torch.cat([key, P["k"].to(key)[None].expand(key.shape[0], -1, -1, -1)], 2).repeat_interleave(g, 1)
        v = torch.cat([value, P["v"].to(value)[None].expand(value.shape[0], -1, -1, -1)], 2).repeat_interleave(g, 1)
        T, S, n = query.shape[2], key.shape[2], P["k"].shape[1]
        logits = (query @ k.transpose(2, 3)).float() * scaling
        if attention_mask is None:
            keep = torch.ones(T, S, dtype=torch.bool, device=query.device).tril(S - T)[None, None]
            real = torch.zeros_like(logits[..., :S]).masked_fill(~keep, -torch.inf)
        elif attention_mask.dtype == torch.bool:
            real = torch.zeros(attention_mask.shape, device=query.device).masked_fill(~attention_mask, -torch.inf)
        else:
            real = attention_mask.float()
        logits[..., :S] += real[..., :S]
        logits[..., S:] -= P["b"]
        # the first few real tokens (the sink) only see themselves; letting them read the prefix rewrites the sink and breaks the model
        seen = (real[..., :S] > -torch.inf).sum(-1, keepdim=True)
        young = (seen > 0) & (seen < MIN_VISIBLE)  # seen == 0: left-padding rows, unused; masking them too gives NaN
        logits[..., S:] = logits[..., S:].masked_fill(young, -torch.inf)
        A = logits.softmax(-1)
        last = A[:, :, -1]  # last query position (real under left padding); diagnostics: mass on all prefix, on the + sentence
        P["mass"] = last[..., S:].sum(-1).mean().item(), last[..., S:S + P["n_pos"]].sum(-1).mean().item()
        return (A.to(v.dtype) @ v).transpose(1, 2).contiguous(), None
    return attention


class _Restore:
    def __init__(self, fn):
        self.fn = fn

    def remove(self):
        self.fn()


def _q_prefix_install(model, cfg, stacked):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    impl = model.config._attn_implementation
    original = ALL_ATTENTION_FUNCTIONS[impl]
    from ..attach import _gather_split_state
    blocks = _get_blocks(model)
    for layer in stacked:
        shared = _gather_split_state(blocks[layer])[0]
        _ACTIVE_PREFIX[id(blocks[layer].self_attn)] = {"k": shared["pk"], "v": shared["pv"], "b": cfg.bias, "n_pos": int(shared["npos"][0])}
    ALL_ATTENTION_FUNCTIONS[impl] = _prefix_attention(original)

    def restore():
        ALL_ATTENTION_FUNCTIONS[impl] = original
        for layer in stacked:
            _ACTIVE_PREFIX.pop(id(blocks[layer].self_attn), None)
    return [*_install_add(model, cfg, {layer: {"x": s["x"]} for layer, s in stacked.items()}, "q"), _Restore(restore)]


@register_config
@dataclass
class QPrefixC(SteeringConfig):
    method: str = "q_prefix"
    bias: float = 8.0  # logit penalty on the persona prefix; set from the C = 0 attention mass in sink_probe/q_prefix_probe


register(type("q_prefix", (), {
    "name": "q_prefix", "extract_from_prompts": True, "cache_intervention": True,
    "extract": staticmethod(_q_prefix_extract), "install": staticmethod(_q_prefix_install),
    "apply": staticmethod(lambda _mod, _x, y, _shared, _stacked, _cfg: y),
}))


# sinkr_rand: control for sinkr_sum. Same residual part; the sink value gets a random unit vector per layer (seeded).
def _sinkr_rand_extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
    out = _sinkr_extract(model, tok, pos_prompts, neg_prompts, cfg, batch_size=batch_size, max_length=max_length)
    g = torch.Generator().manual_seed(cfg.seed)
    for s in out.values():
        if "x" in s["stacked"]:
            s["stacked"]["x"] = _unit_direction(torch.randn(s["stacked"]["x"].shape, generator=g))
    return out


@register_config
@dataclass
class SinkRRandC(SinkRSumC):
    method: str = "sinkr_rand"


register(type("sinkr_rand", (), {
    "name": "sinkr_rand", "extract_from_prompts": True, "cache_intervention": True,
    "extract": staticmethod(_sinkr_rand_extract), "install": staticmethod(_sinkr_install),
    "apply": staticmethod(lambda _mod, _x, y, _shared, _stacked, _cfg: y),
}))


# q_prefix_k: same cache prefix; q* now points at the + persona word's keys instead of being the query mean diff:
#     q*_L,g = mean_{s ∈ " sycophantic"} k_s − mean_{s ∈ " abrasive"} k_s   (post-RoPE, at the tokens where the two sentences differ)
#     q_t (post-RoPE) += C·q̂*_L          so logit(t, s) changes by C·q̂*·k_s, fixed for each prefix key s
def _word_slices(tok):
    a, b = (tok(x, add_special_tokens=False).input_ids for x in PERSONA_SENTENCES)
    i = 0
    while a[i] == b[i]:
        i += 1
    j = 0
    while a[-1 - j] == b[-1 - j]:
        j += 1
    return (i, len(a) - j), (i, len(b) - j)


def _q_prefix_k_extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
    from transformers import AutoTokenizer
    tok = tok or AutoTokenizer.from_pretrained(model.config._name_or_path)
    layers = tuple(range(1, len(_get_blocks(model)))) if cfg.layers is None else tuple(l for l in cfg.layers if l > 0)
    kv, lengths = _persona_kv(model, tok, layers)
    head = tok.apply_chat_template([{"role": "user", "content": PERSONA_SENTENCES[0]}], tokenize=False).split(PERSONA_SENTENCES[0])[0]
    offset = len(tok(head, add_special_tokens=False).input_ids) - 1  # sentence start inside the stored prefix (sink dropped)
    (a0, a1), (b0, b1) = _word_slices(tok)
    out = {}
    for layer in layers:
        k = kv[layer][0]  # [KVH, n⁺ + n⁻, d]
        dq = k[:, offset + a0:offset + a1].mean(1) - k[:, lengths[0] + offset + b0:lengths[0] + offset + b1].mean(1)
        out[layer] = {"shared": {"pk": kv[layer][0], "pv": kv[layer][1], "npos": torch.tensor([lengths[0]])},
                      "stacked": {"x": _unit_direction(dq.flatten()).unsqueeze(0)}}
    return out


def _q_prefix_k_install(model, cfg, stacked):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from ..attach import _gather_split_state
    impl = model.config._attn_implementation
    original = ALL_ATTENTION_FUNCTIONS[impl]
    blocks = _get_blocks(model)
    for layer, s in stacked.items():
        shared = _gather_split_state(blocks[layer])[0]
        attn = blocks[layer].self_attn
        _ACTIVE_PREFIX[id(attn)] = {"k": shared["pk"], "v": shared["pv"], "b": cfg.bias, "n_pos": int(shared["npos"][0]),
                                    "dq": s["x"].sum(0).view(-1, attn.head_dim), "C": cfg.coeff}
    ALL_ATTENTION_FUNCTIONS[impl] = _prefix_attention(original)

    def restore():
        ALL_ATTENTION_FUNCTIONS[impl] = original
        for layer in stacked:
            _ACTIVE_PREFIX.pop(id(blocks[layer].self_attn), None)
    return [_Restore(restore)]


@register_config
@dataclass
class QPrefixKC(QPrefixC):
    method: str = "q_prefix_k"
    bias: float = 4.0  # Qwen3-4B probe: prefix gets ~0.4% of last-token attention at C = 0


@register_config
@dataclass
class QPrefixK0C(QPrefixC):
    method: str = "q_prefix_k0"
    bias: float = 0.0  # prefix visible: ~15% of last-token attention at C = 0


for _name in ("q_prefix_k", "q_prefix_k0"):
    register(type(_name, (), {
        "name": _name, "extract_from_prompts": True, "cache_intervention": True,
        "extract": staticmethod(_q_prefix_k_extract), "install": staticmethod(_q_prefix_k_install),
        "apply": staticmethod(lambda _mod, _x, y, _shared, _stacked, _cfg: y),
    }))


# --- q_slot: split the attention sink into two halves carrying ±v*, and let the query choose between them --------
# Per layer, per KV head g (post-RoPE keys):
#     the real first token (the sink, <|im_start|> at position 0) is hidden from query positions past the first 3
#     slot±: key = k_sink ± ε·u,  value = v_sink ± ν·v̂*,  logit bias −ln 2     (u ⟂ k_sink random unit, ν = ‖v_sink‖)
#     q_t (post-RoPE) += C·u
# At C = 0 the two halves get the sink's weight between them and their values average to v_sink (unchanged up to an
# ε·(q·u) term). As C moves, the sink's attention shifts to one half: the head writes ≈ w_sink·ν·tanh(ε·C·scale)·v̂*.
@torch.no_grad()
def _key_samples(model, tok, prompts, layers, max_length, site="k"):
    """post-RoPE keys (site "k") or queries (site "q", grouped per KV head) of every real token, first token dropped
    -> {layer: [KVH, n, d]} on CPU"""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    blocks, out = _get_blocks(model), {L: [] for L in layers}
    norm = {"k": "k_norm", "q": "q_norm"}[site]
    for text in prompts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length).input_ids.to(next(model.parameters()).device)
        grabbed = {}
        hooks = [getattr(blocks[L].self_attn, norm).register_forward_hook(lambda _m, _i, o, L=L: grabbed.__setitem__(L, o)) for L in layers]
        try:
            model(ids)
        finally:
            for h in hooks:
                h.remove()
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        for L in layers:
            x = grabbed[L].transpose(1, 2)  # [1, heads, T, d]
            cos, sin = model.model.rotary_emb(x, pos)
            x, _ = apply_rotary_pos_emb(x, x, cos, sin)
            x = x[0, :, 1:].float().cpu()
            kvh = blocks[L].self_attn.config.num_key_value_heads
            out[L].append(x.reshape(kvh, -1, x.shape[-1]))  # query heads of one KV group stacked as samples
    return {L: torch.cat(xs, 1) for L, xs in out.items()}


def _q_slot_extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    tok = tok or AutoTokenizer.from_pretrained(model.config._name_or_path)
    v = _mean_diff_site("v")(model, tok, pos_prompts, neg_prompts, cfg, batch_size=batch_size, max_length=max_length)
    blocks, grabbed = _get_blocks(model), {}
    ids = tok(pos_prompts[0], return_tensors="pt", add_special_tokens=False).input_ids[:, :1].to(next(model.parameters()).device)
    hooks = [blocks[L].self_attn.k_norm.register_forward_hook(lambda _m, _i, o, L=L: grabbed.__setitem__(("k", L), o)) for L in v]
    hooks += [blocks[L].self_attn.v_proj.register_forward_hook(lambda _m, _i, o, L=L: grabbed.__setitem__(("v", L), o)) for L in v]
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in hooks:
            h.remove()
    keys = _key_samples(model, tok, pos_prompts[:16] + neg_prompts[:16], tuple(v), max_length)
    queries = _key_samples(model, tok, pos_prompts[:16] + neg_prompts[:16], tuple(v), max_length, site="q")
    out = {}
    for L in v:
        attn = blocks[L].self_attn
        d = attn.head_dim
        k = grabbed[("k", L)].transpose(1, 2)  # [1, KVH, 1, d] pre-RoPE
        cos, sin = model.model.rotary_emb(k, torch.zeros(1, 1, dtype=torch.long, device=k.device))
        k, _ = apply_rotary_pos_emb(k, k, cos, sin)
        k = k[0, :, 0].float().cpu()  # [KVH, d]
        vs = grabbed[("v", L)].view(-1, d).float().cpu()  # [KVH, d]
        # u_g: the direction real keys and queries vary least along (smallest eigenvector of E[kkᵀ]/tr + E[qqᵀ]/tr),
        # made orthogonal to k_sink: C·u in the query barely changes logits to real tokens (small u·k), and the model's
        # own query barely prefers either half at C = 0 (small q·u)
        Mk = torch.einsum("gnd,gne->gde", keys[L], keys[L]) / keys[L].shape[1]
        Mq = torch.einsum("gnd,gne->gde", queries[L], queries[L]) / queries[L].shape[1]
        tr = lambda M: M.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
        M = Mk / tr(Mk) + Mq / tr(Mq)  # [KVH, d, d]
        u = torch.linalg.eigh(M).eigenvectors[..., 0]  # [KVH, d]
        u = u - (u * k).sum(-1, keepdim=True) / (k * k).sum(-1, keepdim=True) * k
        u = u / u.norm(dim=-1, keepdim=True)
        vstar = v[L]["stacked"]["x"][0].view(-1, d).float().cpu()
        if cfg.nu_scale is None:  # per KV head, ν = ‖v_sink‖
            vstar = vstar / vstar.norm(dim=-1, keepdim=True) * vs.norm(dim=-1, keepdim=True)
        else:  # ν_g = nu_scale·‖v̂*_g‖: fully choosing one half writes what sink_value writes at C = nu_scale
            vstar = vstar * cfg.nu_scale
        out[L] = {"shared": {"ksink": k, "vsink": vs, "u": u, "vstar": vstar}, "stacked": {"x": u.flatten().unsqueeze(0)}}
    return out


def _slot_attention(original):
    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        P = _ACTIVE_SLOT.get(id(module))
        if P is None:
            return original(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
        g = module.num_key_value_groups
        ks, vk, u, vs = (P[n].to(query) for n in ("ksink", "vsink", "u", "vstar"))
        slot_k = torch.stack([ks + P["eps"] * u, ks - P["eps"] * u], 1)  # [KVH, 2, d]
        slot_v = torch.stack([vk + vs, vk - vs], 1)
        query = query + P["C"] * u.repeat_interleave(g, 0)[None, :, None, :]
        B = key.shape[0]
        k = torch.cat([key, slot_k[None].expand(B, -1, -1, -1)], 2).repeat_interleave(g, 1)
        v = torch.cat([value, slot_v[None].expand(B, -1, -1, -1)], 2).repeat_interleave(g, 1)
        T, S = query.shape[2], key.shape[2]
        logits = (query @ k.transpose(2, 3)).float() * scaling
        if attention_mask is None:
            real = torch.zeros(T, S, device=query.device).masked_fill(~torch.ones(T, S, dtype=torch.bool, device=query.device).tril(S - T), -torch.inf)[None, None]
        elif attention_mask.dtype == torch.bool:
            real = torch.zeros(attention_mask.shape, device=query.device).masked_fill(~attention_mask, -torch.inf)
        else:
            real = attention_mask.float()
        real = real[..., :S].expand(B, -1, T, S)
        visible = real > -torch.inf  # [B, 1, T, S]
        seen = visible.sum(-1, keepdim=True)
        mature = seen >= MIN_VISIBLE
        first = visible.float().argmax(-1, keepdim=True)  # the first real key = the sink (left padding)
        hide_sink = torch.zeros_like(visible).scatter(-1, first, True) & mature
        logits[..., :S] += real.masked_fill(hide_sink, -torch.inf)
        logits[..., S:] = (logits[..., S:] - math.log(2)).masked_fill(~mature & (seen > 0), -torch.inf)  # seen == 0: padding rows
        A = logits.softmax(-1)
        P["read"] = (A[:, :, -1, S] - A[:, :, -1, S + 1]).mean().item()  # net + half weight at the last query
        return (A.to(v.dtype) @ v).transpose(1, 2).contiguous(), None
    return attention


_ACTIVE_SLOT: dict[int, dict] = {}


def _q_slot_install(model, cfg, stacked):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from ..attach import _gather_split_state
    impl = model.config._attn_implementation
    original = ALL_ATTENTION_FUNCTIONS[impl]
    blocks = _get_blocks(model)
    for L in stacked:
        sh = _gather_split_state(blocks[L])[0]
        _ACTIVE_SLOT[id(blocks[L].self_attn)] = {"ksink": sh["ksink"], "vsink": sh["vsink"], "u": sh["u"], "vstar": sh["vstar"], "eps": cfg.eps, "C": cfg.coeff}
    ALL_ATTENTION_FUNCTIONS[impl] = _slot_attention(original)

    def restore():
        ALL_ATTENTION_FUNCTIONS[impl] = original
        for L in stacked:
            _ACTIVE_SLOT.pop(id(blocks[L].self_attn), None)
    return [_Restore(restore)]


@register_config
@dataclass
class QSlotC(SteeringConfig):
    method: str = "q_slot"
    eps: float = 1.0  # key offset along u; logit gap between the halves = 2·eps·(q·u + C)·scale (0.6B: 0.05 too weak, 5 changes C=0)
    nu_scale: float | None = None


@register_config
@dataclass
class QSlotBigC(QSlotC):
    method: str = "q_slot_big"
    nu_scale: float | None = 12.7  # sink_value's best -C dose on Qwen3-4B dev; ‖v_sink‖ was 1.3-12x smaller than this write


@register_config
@dataclass
class QSlotHugeC(QSlotC):
    method: str = "q_slot_huge"
    nu_scale: float | None = 25.4  # 2x q_slot_big: at the walk's doses only part of the sink weight moves, so the cap binds


for _name in ("q_slot", "q_slot_big", "q_slot_huge"):
    register(type(_name, (), {
        "name": _name, "extract_from_prompts": True, "cache_intervention": True,
        "extract": staticmethod(_q_slot_extract), "install": staticmethod(_q_slot_install),
        "apply": staticmethod(lambda _mod, _x, y, _shared, _stacked, _cfg: y),
    }))
