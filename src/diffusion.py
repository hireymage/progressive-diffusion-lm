"""
Masked (absorbing) diffusion process for discrete token sequences.

How the diffusion process works
---------------------------------
We use a simple MASK-absorbing forward process:

  Forward (corruption):
    Given a clean sequence x_0, sample a mask rate m ~ Uniform(0, 1).
    Each token position is independently masked with probability m:
        x_t[i] = MASK_TOKEN  with probability m
        x_t[i] = x_0[i]     otherwise
    Higher m = more corruption.  m≈1 is "fully noisy", m≈0 is "nearly clean".

  Model:
    p_θ(x_0 | x_t, m).  The model predicts the original token at every
    position given the partially masked sequence.

  Training objective:
    Minimise the reconstruction cross-entropy only at masked positions:
        L = -E_{m,x_t} [ Σ_{i: x_t[i]=MASK} log p_θ(x_0[i] | x_t, m) ]

  Reverse (generation / refinement):
    Start: x_T = all MASK tokens.
    For refinement step i = 1 … T:
      1. Compute predicted logits (using precision schedule[i-1]).
      2. For each currently masked position, compute a probability distribution.
      3. Unmask the top-k highest-confidence positions (argmax confidence).
      4. Sample or argmax the token for those positions.
    After T steps, x_0 is fully revealed.

The "confidence" of a masked position is the probability of the top-predicted
token: max_j softmax(logits[j]).

Progressive precision schedule
-------------------------------
During inference, refinement step i uses precision_schedule[i-1] bits.
During training, we map mask rate m to a step index and apply the
corresponding precision.  This couples higher precision to lower noise,
matching the intuition that fine-grained prediction requires higher precision.
"""

import math
import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


def corrupt_tokens(
    x0: mx.array,          # (B, L) integer token IDs
    mask_rate: float | mx.array,  # scalar or (B,) per-example rate
    mask_token_id: int,
) -> tuple[mx.array, mx.array]:
    """
    Apply the forward diffusion (masking) process.

    Returns
    -------
    x_t : (B, L) — masked token sequence
    mask : (B, L) bool — True where position was masked
    """
    B, L = x0.shape
    if isinstance(mask_rate, float):
        mask_rate = mx.full((B,), mask_rate)
    # mask_rate: (B,) → broadcast to (B, L)
    noise = mx.random.uniform(shape=(B, L))
    mask = noise < mask_rate[:, None]  # (B, L) bool
    x_t = mx.where(mask, mx.full_like(x0, mask_token_id), x0)
    return x_t, mask


def mask_rate_to_step(mask_rate: float, n_steps: int) -> int:
    """Map a continuous mask rate in [0,1] to a discrete step index [0, T-1]."""
    # step 0 → highest noise (mask_rate near 1.0)
    # step T-1 → lowest noise (mask_rate near 0.0)
    step = int((1.0 - mask_rate) * n_steps)
    return max(0, min(n_steps - 1, step))


def compute_loss(
    model,
    x0: mx.array,           # (B, L) clean token IDs
    mask_token_id: int,
    precision_schedule: list[int],
    n_steps: int,
) -> mx.array:
    """
    Compute masked-diffusion training loss for one batch.

    Procedure:
      1. Sample mask rates uniformly per example.
      2. Apply masking (forward diffusion).
      3. Map each example's mask rate to a precision (via schedule).
         Because all examples in the batch must use the same `bits` (for
         simplicity), we use the mean step of the batch.
      4. Run model forward.
      5. Cross-entropy at masked positions.
    """
    B, L = x0.shape

    # Sample mask rates
    mask_rates = mx.random.uniform(low=0.1, high=1.0, shape=(B,))

    # Corrupt
    x_t, mask = corrupt_tokens(x0, mask_rates, mask_token_id)

    # Step index for precision lookup: use mean mask rate of batch
    mean_rate = float(mask_rates.mean())
    step_idx = mask_rate_to_step(mean_rate, n_steps)
    bits = precision_schedule[step_idx]

    # Set model precision
    model.set_bits(bits)

    # Step fraction fed to model (1.0 = fully noisy, 0.0 = clean)
    step_frac = mask_rates  # we use mask_rate directly as the step fraction

    logits = model(x_t, step_frac)  # (B, L, vocab_size)

    # Cross-entropy loss only at masked positions
    # logits: (B, L, V), targets: (B, L)
    V = logits.shape[-1]
    flat_logits = logits.reshape(-1, V)
    flat_targets = x0.reshape(-1)
    flat_mask = mask.reshape(-1)

    log_probs = nn.log_softmax(flat_logits, axis=-1)
    token_loss = -log_probs[mx.arange(flat_logits.shape[0]), flat_targets]

    # Average over masked positions
    n_masked = flat_mask.sum().astype(mx.float32)
    loss = (token_loss * flat_mask).sum() / mx.maximum(n_masked, mx.array(1.0))
    return loss


@mx.compile
def _generate_step(
    model,
    x: mx.array,
    step_frac: mx.array,
    mask_token_id: int,
    n_to_unmask: int,
) -> mx.array:
    """One refinement step: predict + unmask the most confident positions."""
    B, L = x.shape
    logits = model(x, step_frac)  # (B, L, V)
    probs = nn.softmax(logits, axis=-1)  # (B, L, V)

    # Confidence = max token probability at each position
    confidence = probs.max(axis=-1)   # (B, L)
    pred_ids = probs.argmax(axis=-1)  # (B, L)

    # Only consider currently masked positions
    is_masked = (x == mask_token_id)  # (B, L) bool

    # Set confidence of already-unmasked positions to -inf so they're not selected
    confidence = mx.where(is_masked, confidence, mx.full_like(confidence, -1e9))

    # Unmask top-n most confident masked positions per sequence
    # (sort descending by confidence, take top n)
    sorted_indices = mx.argsort(-confidence, axis=-1)  # (B, L) descending

    new_x = x
    for b in range(B):
        indices_to_unmask = sorted_indices[b, :n_to_unmask]
        for idx_pos in range(n_to_unmask):
            pos = int(sorted_indices[b, idx_pos])
            if confidence[b, pos] > -1e8:   # is a real masked position
                new_x = mx.array(new_x.tolist())  # materialize for index write
                tokens = new_x[b].tolist()
                tokens[pos] = int(pred_ids[b, pos])
                new_x = mx.array(
                    new_x[:b].tolist() + [tokens] + new_x[b+1:].tolist()
                )
    return new_x


def generate(
    model,
    seq_len: int,
    mask_token_id: int,
    precision_schedule: list[int],
    n_steps: int | None = None,
    temperature: float = 1.0,
    batch_size: int = 1,
) -> mx.array:
    """
    Generate token sequences via progressive masked-diffusion refinement.

    Parameters
    ----------
    model : DiffusionLM
    seq_len : length of sequence to generate
    mask_token_id : the [MASK] token ID
    precision_schedule : list of bits per step (length = n_steps)
    n_steps : number of refinement steps (default: len(precision_schedule))
    temperature : softmax temperature for sampling (1.0 = no change)
    batch_size : number of sequences to generate in parallel

    Returns
    -------
    token_ids : (batch_size, seq_len) integer array
    """
    if n_steps is None:
        n_steps = len(precision_schedule)

    # Start from all-masked sequence
    x = mx.full((batch_size, seq_len), mask_token_id, dtype=mx.int32)

    tokens_per_step = max(1, seq_len // n_steps)

    for step_i in range(n_steps):
        bits = precision_schedule[step_i]
        model.set_bits(bits)

        # step_frac decreases from 1.0 (fully noisy) to near 0.0 (nearly clean)
        current_frac = 1.0 - step_i / n_steps
        step_frac = mx.full((batch_size,), current_frac)

        # How many tokens to reveal at this step
        target_unmasked = int((step_i + 1) / n_steps * seq_len)
        current_unmasked = int((x != mask_token_id).sum()) // batch_size
        n_to_reveal = max(0, target_unmasked - current_unmasked)

        if n_to_reveal == 0:
            continue

        # Forward pass
        logits = model(x, step_frac)  # (B, L, V)

        if temperature != 1.0:
            logits = logits / temperature

        probs = nn.softmax(logits.astype(mx.float32), axis=-1)
        pred_ids = probs.argmax(axis=-1)         # (B, L)
        confidence = probs.max(axis=-1)          # (B, L)

        is_masked = (x == mask_token_id)

        # Evaluate to numpy for per-sequence index manipulation
        mx.eval(x, confidence, pred_ids, is_masked)

        x_np = x.tolist()
        conf_np = confidence.tolist()
        pred_np = pred_ids.tolist()
        masked_np = is_masked.tolist()

        for b in range(batch_size):
            # Collect (confidence, position) for masked positions
            candidates = [
                (conf_np[b][p], p)
                for p in range(seq_len)
                if masked_np[b][p]
            ]
            # Sort descending by confidence, pick top n_to_reveal
            candidates.sort(reverse=True)
            for _, pos in candidates[:n_to_reveal]:
                x_np[b][pos] = pred_np[b][pos]

        x = mx.array(x_np, dtype=mx.int32)
        mx.eval(x)

    # Ensure any remaining masked tokens are filled with model's best guess
    still_masked = (x == mask_token_id)
    if bool(still_masked.any()):
        step_frac = mx.zeros((batch_size,))
        model.set_bits(precision_schedule[-1])
        logits = model(x, step_frac)
        mx.eval(logits)
        pred_ids = logits.argmax(axis=-1)
        x = mx.where(still_masked, pred_ids, x)
        mx.eval(x)

    return x


def generate_incremental(
    model,
    seq_len: int,
    mask_token_id: int,
    precision_schedule: list[int],
    n_steps: int | None = None,
    temperature: float = 1.0,
    batch_size: int = 1,
    delta_weight: float = 1.0,
) -> mx.array:
    """
    Generate via progressive refinement using the incremental forward API.

    This is the Phase 2 variant of ``generate`` that uses
    ``model.forward_incremental`` with an ``IncrementalCache`` to compute
        y_next = y_prev + delta_weight * Δ
    at each refinement step.

    When delta_weight=1.0 the output is mathematically identical to the
    standard ``generate`` function (parity guarantee).  When delta_weight<1.0
    the residual is attenuated, which may improve robustness at the cost of
    slightly slower convergence.

    Parameters
    ----------
    model : DiffusionLM
    seq_len : length of sequence to generate
    mask_token_id : the [MASK] token ID
    precision_schedule : list of bits per step
    n_steps : number of refinement steps (default: len(precision_schedule))
    temperature : softmax temperature
    batch_size : number of sequences to generate in parallel
    delta_weight : residual scaling factor (1.0 = exact parity)

    Returns
    -------
    token_ids : (batch_size, seq_len) integer array
    """
    from .model import IncrementalCache

    if n_steps is None:
        n_steps = len(precision_schedule)

    x = mx.full((batch_size, seq_len), mask_token_id, dtype=mx.int32)
    cache: IncrementalCache | None = None

    for step_i in range(n_steps):
        bits = precision_schedule[step_i]
        model.set_bits(bits)

        current_frac = 1.0 - step_i / n_steps
        step_frac = mx.full((batch_size,), current_frac)

        target_unmasked = int((step_i + 1) / n_steps * seq_len)
        current_unmasked = int((x != mask_token_id).sum()) // batch_size
        n_to_reveal = max(0, target_unmasked - current_unmasked)

        if n_to_reveal == 0:
            continue

        logits, cache = model.forward_incremental(x, step_frac, cache, delta_weight)

        if temperature != 1.0:
            logits = logits / temperature

        probs = nn.softmax(logits.astype(mx.float32), axis=-1)
        pred_ids = probs.argmax(axis=-1)
        confidence = probs.max(axis=-1)
        is_masked = (x == mask_token_id)

        mx.eval(x, confidence, pred_ids, is_masked)

        x_np = x.tolist()
        conf_np = confidence.tolist()
        pred_np = pred_ids.tolist()
        masked_np = is_masked.tolist()

        for b in range(batch_size):
            candidates = [
                (conf_np[b][p], p)
                for p in range(seq_len)
                if masked_np[b][p]
            ]
            candidates.sort(reverse=True)
            for _, pos in candidates[:n_to_reveal]:
                x_np[b][pos] = pred_np[b][pos]

        x = mx.array(x_np, dtype=mx.int32)
        mx.eval(x)

    still_masked = (x == mask_token_id)
    if bool(still_masked.any()):
        step_frac = mx.zeros((batch_size,))
        model.set_bits(precision_schedule[-1])
        logits, _ = model.forward_incremental(x, step_frac, cache, delta_weight)
        mx.eval(logits)
        pred_ids = logits.argmax(axis=-1)
        x = mx.where(still_masked, pred_ids, x)
        mx.eval(x)

    return x


def generate_with_early_exit(
    model,
    seq_len: int,
    mask_token_id: int,
    precision_schedule: list[int],
    confidence_threshold: float = 0.95,
    n_steps: int | None = None,
    temperature: float = 1.0,
    batch_size: int = 1,
    min_steps: int = 1,
    use_incremental: bool = False,
    delta_weight: float = 1.0,
) -> tuple[mx.array, int]:
    """
    Generate with early-exit: stop refinement once mean confidence exceeds
    a threshold, saving compute on 'easy' sequences.

    Parameters
    ----------
    model : DiffusionLM
    seq_len : length of sequence to generate
    mask_token_id : the MASK token ID
    precision_schedule : list of bits per step
    confidence_threshold : mean top-1 confidence above which we stop early
    n_steps : max number of refinement steps (default: len(schedule))
    temperature : softmax temperature
    batch_size : number of sequences to generate in parallel
    min_steps : minimum steps before early-exit is allowed (default 1)
    use_incremental : if True, use forward_incremental instead of standard forward
    delta_weight : residual scaling for incremental mode (1.0 = parity)

    Returns
    -------
    token_ids : (batch_size, seq_len) integer array
    steps_used : int - actual number of refinement steps executed
    """
    from .model import IncrementalCache

    if n_steps is None:
        n_steps = len(precision_schedule)

    x = mx.full((batch_size, seq_len), mask_token_id, dtype=mx.int32)
    cache: IncrementalCache | None = None
    steps_used = 0

    for step_i in range(n_steps):
        bits = precision_schedule[step_i]
        model.set_bits(bits)

        current_frac = 1.0 - step_i / n_steps
        step_frac = mx.full((batch_size,), current_frac)

        target_unmasked = int((step_i + 1) / n_steps * seq_len)
        current_unmasked = int((x != mask_token_id).sum()) // batch_size
        n_to_reveal = max(0, target_unmasked - current_unmasked)

        if n_to_reveal == 0:
            continue

        if use_incremental:
            logits, cache = model.forward_incremental(x, step_frac, cache, delta_weight)
        else:
            logits = model(x, step_frac)

        if temperature != 1.0:
            logits = logits / temperature

        probs = nn.softmax(logits.astype(mx.float32), axis=-1)
        pred_ids = probs.argmax(axis=-1)
        confidence = probs.max(axis=-1)
        is_masked = (x == mask_token_id)

        mx.eval(x, confidence, pred_ids, is_masked)

        # Early-exit check: mean confidence of masked positions
        conf_np = confidence.tolist()
        masked_np = is_masked.tolist()
        masked_confs = [
            conf_np[b][p]
            for b in range(batch_size)
            for p in range(seq_len)
            if masked_np[b][p]
        ]
        mean_conf = sum(masked_confs) / max(len(masked_confs), 1)

        # Reveal tokens
        x_np = x.tolist()
        pred_np = pred_ids.tolist()
        for b in range(batch_size):
            candidates = [
                (conf_np[b][p], p)
                for p in range(seq_len)
                if masked_np[b][p]
            ]
            candidates.sort(reverse=True)
            for _, pos in candidates[:n_to_reveal]:
                x_np[b][pos] = pred_np[b][pos]

        x = mx.array(x_np, dtype=mx.int32)
        mx.eval(x)
        steps_used += 1

        # Early exit: if confidence is high enough and we've done min_steps
        if steps_used >= min_steps and mean_conf >= confidence_threshold:
            break

    # Fill any remaining masked tokens with model's best guess
    still_masked = (x == mask_token_id)
    if bool(still_masked.any()):
        step_frac = mx.zeros((batch_size,))
        model.set_bits(precision_schedule[-1])
        if use_incremental:
            logits, _ = model.forward_incremental(x, step_frac, cache, delta_weight)
        else:
            logits = model(x, step_frac)
        mx.eval(logits)
        pred_ids = logits.argmax(axis=-1)
        x = mx.where(still_masked, pred_ids, x)
        mx.eval(x)

    return x, steps_used