"""Smoke: verify calibrate_iso_kl with log-log secant converges fast."""
import torch
import steering_lite as sl
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    mid = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mid).eval()

    pos = ["the cat sat on the mat"] * 8
    neg = ["the dog ran in the park"] * 8
    cfg = sl.MeanDiffC(layers=(1,), coeff=1.0, dtype=torch.float32, seed=0)
    vectors = sl.train(model, tok, pos, neg, cfg, batch_size=4, max_length=32)
    prompts = [tok("hello world", return_tensors="pt").input_ids[0] for _ in range(2)]
    coeff, hist = sl.calibrate_iso_kl(
        model, prompts, cfg, vectors,
        target_kl=0.005, target_stat="kl_p95", T=8,
        bracket=(0.01, 4.0),
        eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
        do_sample=False, device="cpu",
    )
    print(f"FINAL coeff={coeff:.4f} iters={len(hist)}")
    for h in hist:
        print(f"  c={h['coeff']:.4f} p95={h['kl_p95']:.4f}")
    assert len(hist) <= 12, f"too many iters: {len(hist)}"
    # at least one eval landed near target
    near = [h for h in hist if abs(h["kl_p95"] - 0.005) < 0.002]
    assert near, f"never came near target; final p95={hist[-1]['kl_p95']:.4f}"
    # secant should converge inside ~6 iters on this monotone curve
    assert len(hist) <= 6, f"secant should be fast; got {len(hist)} iters"
    print(f"SMOKE OK ({len(hist)} iters)")


if __name__ == "__main__":
    main()
