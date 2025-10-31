from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


class SequenceEncoder(eqx.Module):
    """Single-direction GRU encoder that respects padding masks."""

    gru: eqx.nn.GRUCell

    def __init__(self, *, hidden_dim: int, key: jax.Array) -> None:
        self.gru = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=key)

    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def step(carry, data):
            x, is_valid = data
            is_valid = jnp.asarray(is_valid, dtype=bool)
            proposed = self.gru(x, carry)
            next_carry = jnp.where(is_valid, proposed, carry)
            return next_carry, next_carry

        init_state = jnp.zeros(inputs.shape[-1], dtype=inputs.dtype)
        final_state, outputs = jax.lax.scan(step, init_state, (inputs, mask))
        return outputs, final_state


class BidirectionalEncoder(eqx.Module):
    forward: SequenceEncoder
    backward: SequenceEncoder
    merger: eqx.nn.Linear

    def __init__(self, *, hidden_dim: int, key: jax.Array) -> None:
        key_f, key_b, key_m = jax.random.split(key, 3)
        self.forward = SequenceEncoder(hidden_dim=hidden_dim, key=key_f)
        self.backward = SequenceEncoder(hidden_dim=hidden_dim, key=key_b)
        self.merger = eqx.nn.Linear(hidden_dim * 2, hidden_dim, use_bias=True, key=key_m)

    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        mask_bool = jnp.asarray(mask, dtype=bool)
        outputs_f, state_f = self.forward(inputs, mask_bool)

        rev_inputs = jnp.flip(inputs, axis=0)
        rev_mask = jnp.flip(mask_bool, axis=0)
        outputs_b, state_b = self.backward(rev_inputs, rev_mask)
        outputs_b = jnp.flip(outputs_b, axis=0)

        concat = jnp.concatenate([outputs_f, outputs_b], axis=-1)
        combined = jax.vmap(self.merger)(concat)
        global_state = (state_f + state_b) / 2
        return combined, global_state


class StackedBidirectionalEncoder(eqx.Module):
    layers: tuple[BidirectionalEncoder, ...]

    def __init__(self, *, hidden_dim: int, depth: int, key: jax.Array) -> None:
        keys = jax.random.split(key, depth)
        self.layers = tuple(BidirectionalEncoder(hidden_dim=hidden_dim, key=k) for k in keys)

    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        outputs = inputs
        for layer in self.layers:
            outputs, _ = layer(outputs, mask)
            outputs = jnp.tanh(outputs)

        mask_f = jnp.asarray(mask, dtype=outputs.dtype)
        denom = jnp.maximum(jnp.sum(mask_f), 1.0)
        global_state = jnp.sum(outputs * mask_f[:, None], axis=0) / denom
        return outputs, global_state


@dataclasses.dataclass
class MaskDiffusionSchedule:
    """Keep-probability schedule for masked discrete diffusion."""

    keep_probs: jnp.ndarray

    def __post_init__(self) -> None:
        if self.keep_probs.ndim != 1:
            raise ValueError("keep_probs must be 1D")
        if self.keep_probs.shape[0] < 2:
            raise ValueError("keep_probs must contain at least two levels")
        if not jnp.all(self.keep_probs[:-1] >= self.keep_probs[1:]):
            raise ValueError("keep_probs must be monotonically non-increasing")

    @property
    def n_levels(self) -> int:
        return int(self.keep_probs.shape[0])


def default_mask_schedule(n_steps: int, *, min_keep: float = 0.2) -> MaskDiffusionSchedule:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    min_keep = max(min_keep, 0.0)
    max_keep = 1.0
    keep_values = jnp.linspace(max_keep, min_keep, num=n_steps + 1)
    return MaskDiffusionSchedule(keep_probs=keep_values)


def diffuse_tokens_pair(
    key: jax.Array,
    clean_tokens: jnp.ndarray,
    *,
    schedule: MaskDiffusionSchedule,
    t: int,
    pad_id: int,
    mask_id: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    max_index = schedule.n_levels - 2
    t = jnp.clip(jnp.asarray(t, dtype=jnp.int32), 0, max_index)
    noise = jax.random.uniform(key, clean_tokens.shape)
    keep_prob_t = schedule.keep_probs[t]
    keep_prob_next = schedule.keep_probs[t + 1]
    keep_t = noise < keep_prob_t
    keep_next = noise < keep_prob_next
    pad_mask = clean_tokens == pad_id
    keep_t = jnp.logical_or(keep_t, pad_mask)
    keep_next = jnp.logical_or(keep_next, pad_mask)
    x_t = jnp.where(keep_t, clean_tokens, mask_id)
    x_next = jnp.where(keep_next, clean_tokens, mask_id)
    return x_t, x_next


class TextDiffusionModel(eqx.Module):
    """Shared weights for masked text diffusion with cross-attention helpers."""

    prev_embed: jnp.ndarray
    input_embed: jnp.ndarray
    step_embed: jnp.ndarray
    decoder: jnp.ndarray
    bias: jnp.ndarray
    encoder: StackedBidirectionalEncoder
    position_embed: jnp.ndarray
    month_head: eqx.nn.Linear
    day_head: eqx.nn.Linear
    year_head: eqx.nn.Linear
    digit_token_ids: np.ndarray = eqx.field(static=True)
    hyphen_token_id: int = eqx.field(static=True)
    start_token_id: int = eqx.field(static=True)
    end_token_id: int = eqx.field(static=True)
    year_digits_table: np.ndarray = eqx.field(static=True)
    month_digits_table: np.ndarray = eqx.field(static=True)
    day_digits_table: np.ndarray = eqx.field(static=True)
    target_vocab_size: int = eqx.field(static=True)
    input_vocab_size: int = eqx.field(static=True)
    target_mask_id: int = eqx.field(static=True)
    target_pad_id: int = eqx.field(static=True)
    input_pad_id: int = eqx.field(static=True)
    sequence_length: int = eqx.field(static=True)
    year_offset: int = eqx.field(static=True)
    year_classes: int = eqx.field(static=True)
    iso_length: int = eqx.field(static=True)

    @staticmethod
    def init(
        key: jax.Array,
        *,
        input_vocab_size: int,
        target_vocab_size: int,
        hidden_dim: int,
        n_steps: int,
        target_mask_id: int,
        target_pad_id: int,
        input_pad_id: int,
        sequence_length: int,
        year_offset: int,
        year_classes: int,
        digit_token_ids: jnp.ndarray,
        hyphen_token_id: int,
        start_token_id: int,
        end_token_id: int,
        year_digits_table: jnp.ndarray,
        month_digits_table: jnp.ndarray,
        day_digits_table: jnp.ndarray,
        iso_length: int,
        encoder_layers: int,
    ) -> "TextDiffusionModel":
        key_prev, key_input, key_step, key_dec, key_bias, key_enc, key_pos, key_month, key_day, key_year = jax.random.split(
            key, 10
        )
        scale = 0.02
        prev_embed = jax.random.normal(key_prev, (target_vocab_size, hidden_dim)) * scale
        input_embed = jax.random.normal(key_input, (input_vocab_size, hidden_dim)) * scale
        step_embed = jax.random.normal(key_step, (n_steps, hidden_dim)) * scale
        decoder = jax.random.normal(key_dec, (hidden_dim, target_vocab_size)) * scale
        bias = jax.random.normal(key_bias, (target_vocab_size,)) * scale
        encoder = StackedBidirectionalEncoder(hidden_dim=hidden_dim, depth=encoder_layers, key=key_enc)
        position_embed = jax.random.normal(key_pos, (sequence_length, hidden_dim)) * scale
        month_head = eqx.nn.Linear(hidden_dim, 12, use_bias=True, key=key_month)
        day_head = eqx.nn.Linear(hidden_dim, 31, use_bias=True, key=key_day)
        year_head = eqx.nn.Linear(hidden_dim, year_classes, use_bias=True, key=key_year)
        return TextDiffusionModel(
            prev_embed=prev_embed,
            input_embed=input_embed,
            step_embed=step_embed,
            decoder=decoder,
            bias=bias,
            encoder=encoder,
            position_embed=position_embed,
            month_head=month_head,
            day_head=day_head,
            year_head=year_head,
            digit_token_ids=tuple(int(x) for x in digit_token_ids),
            hyphen_token_id=int(hyphen_token_id),
            start_token_id=int(start_token_id),
            end_token_id=int(end_token_id),
            year_digits_table=tuple(tuple(int(x) for x in row) for row in year_digits_table),
            month_digits_table=tuple(tuple(int(x) for x in row) for row in month_digits_table),
            day_digits_table=tuple(tuple(int(x) for x in row) for row in day_digits_table),
            target_vocab_size=target_vocab_size,
            input_vocab_size=input_vocab_size,
            target_mask_id=target_mask_id,
            target_pad_id=target_pad_id,
            input_pad_id=input_pad_id,
            sequence_length=sequence_length,
            year_offset=year_offset,
            year_classes=year_classes,
            iso_length=iso_length,
        )

    def _encode_inputs(self, input_tokens: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        input_vec = self.input_embed[input_tokens]
        pos = self.position_embed[: input_vec.shape[0]]
        input_vec = input_vec + pos
        valid_mask = input_tokens != self.input_pad_id
        encoder_states, global_state = self.encoder(input_vec, valid_mask)
        mask_f = valid_mask.astype(input_vec.dtype)
        encoder_states = encoder_states * mask_f[:, None]
        return encoder_states, global_state, mask_f

    def component_logits(self, input_tokens: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, global_state, _ = self._encode_inputs(input_tokens)
        month_logits = self.month_head(global_state)
        day_logits = self.day_head(global_state)
        year_logits = self.year_head(global_state)
        return month_logits, day_logits, year_logits

    def _component_log_probs(self, input_tokens: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        month_logits, day_logits, year_logits = self.component_logits(input_tokens)
        return (
            jax.nn.log_softmax(month_logits),
            jax.nn.log_softmax(day_logits),
            jax.nn.log_softmax(year_logits),
        )

    @staticmethod
    def _digits_from_components(log_probs: jnp.ndarray, digits_table: jnp.ndarray) -> jnp.ndarray:
        digits_table = jnp.asarray(digits_table, dtype=jnp.int32)
        digits = jnp.arange(10, dtype=jnp.int32)

        def per_position(digits_pos):
            def per_digit(d):
                mask = digits_pos == d
                masked = jnp.where(mask, log_probs, -jnp.inf)
                return jsp.special.logsumexp(masked)

            return jax.vmap(per_digit)(digits)

        result = jax.vmap(per_position)(digits_table.T)
        return jnp.where(jnp.isneginf(result), jnp.full_like(result, -1e9), result)

    def compute_logits(
        self,
        prev_tokens: jnp.ndarray,
        input_tokens: jnp.ndarray,
        step_index: int,
    ) -> jnp.ndarray:
        del prev_tokens, step_index
        dtype = self.decoder.dtype
        seq_len = input_tokens.shape[0]
        logits = jnp.full((seq_len, self.target_vocab_size), -1e9, dtype=dtype)

        digit_ids = jnp.asarray(self.digit_token_ids, dtype=jnp.int32)

        month_lp, day_lp, year_lp = self._component_log_probs(input_tokens)
        year_digit_lp = self._digits_from_components(year_lp, self.year_digits_table)
        month_digit_lp = self._digits_from_components(month_lp, self.month_digits_table)
        day_digit_lp = self._digits_from_components(day_lp, self.day_digits_table)

        start_pos = 0
        logits = logits.at[start_pos, self.start_token_id].set(0.0)

        iso_positions = list(range(1, 1 + self.iso_length))

        for i in range(4):
            pos = iso_positions[i]
            logits = logits.at[pos, digit_ids].set(year_digit_lp[i])

        hyphen_positions = [iso_positions[4], iso_positions[7]]
        for pos in hyphen_positions:
            logits = logits.at[pos, self.hyphen_token_id].set(0.0)

        for j in range(2):
            pos = iso_positions[5 + j]
            logits = logits.at[pos, digit_ids].set(month_digit_lp[j])

        for j in range(2):
            pos = iso_positions[8 + j]
            logits = logits.at[pos, digit_ids].set(day_digit_lp[j])

        end_pos = 1 + self.iso_length
        if end_pos < seq_len:
            logits = logits.at[end_pos, self.end_token_id].set(0.0)
        if end_pos + 1 < seq_len:
            logits = logits.at[end_pos + 1 :, self.target_pad_id].set(0.0)

        return logits
