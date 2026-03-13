#!/usr/bin/env python3
"""Test whether the local machine can run NanoGPT-style training.

Checks:
1. MPS (Apple Silicon GPU) availability
2. Minimal training loop throughput on MPS vs CPU
3. Compatibility issues with train.py operations (torch.compile, etc.)
4. Docker GPU passthrough feasibility

Usage:
    python scripts/test_local_training.py
"""

import platform
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_hardware():
    """Report hardware capabilities."""
    print("=" * 60)
    print("HARDWARE CHECK")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")

    # MPS check
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    print(f"\nMPS available: {mps_available}")
    print(f"MPS built: {mps_built}")

    # CUDA check (unlikely on Mac but check anyway)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if mps_available:
        print("\n→ Apple Silicon GPU detected — can use MPS backend")
    elif cuda_available:
        print(f"\n→ CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n→ No GPU acceleration available, CPU only")

    return "mps" if mps_available else ("cuda" if cuda_available else "cpu")


def benchmark_training(device_name: str, n_steps: int = 100):
    """Run a minimal GPT-like training loop and measure throughput."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING BENCHMARK ({device_name.upper()})")
    print(f"{'=' * 60}")

    device = torch.device(device_name)

    # Minimal GPT-like model
    vocab_size = 8192
    n_embd = 256
    n_head = 4
    n_layer = 4
    seq_len = 512
    batch_size = 8

    class MiniGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, n_embd)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=n_embd, nhead=n_head,
                    dim_feedforward=n_embd * 4, batch_first=True,
                    dropout=0.0,
                )
                for _ in range(n_layer)
            ])
            self.ln = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size, bias=False)

        def forward(self, x):
            h = self.wte(x)
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            for block in self.blocks:
                h = block(h, src_mask=mask, is_causal=True)
            return self.head(self.ln(h))

    model = MiniGPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params, {n_layer} layers, {n_embd} dim")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(3):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), x[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Benchmark
    if device_name == "mps":
        torch.mps.synchronize()
    elif device_name == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    total_tokens = 0
    for step in range(n_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), x[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += batch_size * seq_len

    if device_name == "mps":
        torch.mps.synchronize()
    elif device_name == "cuda":
        torch.cuda.synchronize()

    dt = time.time() - t0
    tok_per_sec = total_tokens / dt

    print(f"Steps: {n_steps}")
    print(f"Time: {dt:.1f}s")
    print(f"Throughput: {tok_per_sec:,.0f} tok/sec")
    print(f"Final loss: {loss.item():.4f}")

    return tok_per_sec


def check_torch_compile(device_name: str):
    """Test if torch.compile works on this device."""
    print(f"\n{'=' * 60}")
    print("TORCH.COMPILE CHECK")
    print("=" * 60)

    device = torch.device(device_name)
    model = nn.Linear(64, 64).to(device)

    try:
        compiled = torch.compile(model)
        x = torch.randn(8, 64, device=device)
        y = compiled(x)
        print(f"torch.compile: WORKS on {device_name}")
        return True
    except Exception as e:
        print(f"torch.compile: FAILED on {device_name} — {e}")
        print("→ train.py uses torch.compile; would need fallback for local training")
        return False


def check_docker():
    """Check Docker availability and GPU passthrough."""
    print(f"\n{'=' * 60}")
    print("DOCKER CHECK")
    print("=" * 60)

    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Docker: {result.stdout.strip()}")
        else:
            print("Docker: not available")
            return
    except FileNotFoundError:
        print("Docker: not installed")
        return

    print("\nGPU passthrough on macOS:")
    print("  - NVIDIA GPU passthrough: NOT supported (macOS has no NVIDIA drivers)")
    print("  - MPS passthrough: NOT supported (MPS is not exposed to containers)")
    print("  - Docker Desktop for Mac runs Linux VMs — no GPU access")
    print("  → Docker training on Mac must use CPU only (not practical)")
    print("  → For GPU training, use the remote VM directly")


def check_compatibility():
    """Check compatibility with train.py operations."""
    print(f"\n{'=' * 60}")
    print("TRAIN.PY COMPATIBILITY")
    print("=" * 60)

    issues = []

    # Check bfloat16 support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        try:
            x = torch.randn(4, 4, dtype=torch.bfloat16, device=device)
            print("bfloat16 on MPS: SUPPORTED")
        except Exception as e:
            print(f"bfloat16 on MPS: NOT SUPPORTED — {e}")
            issues.append("bfloat16 not supported on MPS")

        # Check SDPA
        try:
            q = torch.randn(1, 4, 8, 32, device=device)
            k = torch.randn(1, 4, 8, 32, device=device)
            v = torch.randn(1, 4, 8, 32, device=device)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print("SDPA (scaled_dot_product_attention): SUPPORTED")
        except Exception as e:
            print(f"SDPA: NOT SUPPORTED — {e}")
            issues.append("SDPA not supported")

        # Check RMSNorm
        try:
            x = torch.randn(4, 64, device=device)
            y = F.rms_norm(x, (64,))
            print("rms_norm: SUPPORTED")
        except Exception as e:
            print(f"rms_norm: NOT SUPPORTED — {e}")
            issues.append("rms_norm not supported")

    if issues:
        print(f"\n→ {len(issues)} compatibility issues found")
        print("  train.py would need modifications for local MPS training")
    else:
        print("\n→ All key operations supported!")

    return issues


def main():
    device = check_hardware()

    # Benchmark on available device
    tok_sec = benchmark_training(device)

    # Also benchmark CPU for comparison
    if device != "cpu":
        cpu_tok_sec = benchmark_training("cpu", n_steps=20)
        speedup = tok_sec / cpu_tok_sec
        print(f"\n→ {device.upper()} speedup over CPU: {speedup:.1f}x")

    # VM comparison
    # RTX PRO 6000 does ~2.5M tok/sec with full train.py
    vm_tok_sec = 2_500_000
    print(f"\n→ VM (RTX PRO 6000) estimate: ~{vm_tok_sec:,} tok/sec")
    print(f"→ Local ({device.upper()}): {tok_sec:,.0f} tok/sec")
    print(f"→ Local is ~{vm_tok_sec / tok_sec:.0f}x slower than VM")

    check_torch_compile(device)
    check_compatibility()
    check_docker()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    if device == "mps":
        print("✓ MPS GPU available — can run training locally")
        print("  Expect ~10-50x slower than the RTX PRO 6000 VM")
        print("  Useful for development/debugging, not production runs")
        print("  Docker GPU passthrough NOT possible on macOS")
    elif device == "cuda":
        print("✓ CUDA GPU available — can run training locally")
    else:
        print("✗ No GPU — local training would be CPU-only (very slow)")


if __name__ == "__main__":
    main()
