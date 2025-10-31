from __future__ import annotations

import dataclasses
import math
import random
from typing import List, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import numpy as np
from jaxtyping import Array

from thrml.block_sampling import SamplingSchedule, sample_states

from .data import (
    CharacterTokenizer,
    DateExpressionGenerator,
    build_default_tokenizer,
    build_iso_tokenizer,
    generate_date_dataset,
)
from .model import MaskDiffusionSchedule, TextDiffusionModel, default_mask_schedule, diffuse_tokens_pair
from .program import TextDiffusionProgramSpec, build_text_diffusion_program


@dataclasses.dataclass
class DiffusionTrainingConfig:
    dataset_size: int = 5000
    max_length: int = 64
    n_steps: int = 4
    hidden_dim: int = 128
    encoder_layers: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 10
    min_keep: float = 0.2
    seed: int = 0
    auxiliary_weight: float = 0.1
    curriculum_schedule: List[tuple[int, List[int]]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TrainingMetrics:
    epoch_losses: List[float]
    epoch_accuracies: List[float]


@dataclasses.dataclass
class DateDiffusionPipeline:
    input_tokenizer: CharacterTokenizer
    target_tokenizer: CharacterTokenizer
    model: TextDiffusionModel
    schedule: MaskDiffusionSchedule
    program_spec: TextDiffusionProgramSpec

    def sample(
        self,
        text: str,
        *,
        key: Array | None = None,
        warmup: int | None = None,
        steps_per_sample: int = 1,
    ) -> str:
        if key is None:
            key = jax.random.PRNGKey(0)
        encoded = jnp.array(self.input_tokenizer.encode(text), dtype=jnp.int32)
        seq_len = encoded.shape[0]
        mask_state = jnp.full((seq_len,), self.target_tokenizer.mask_id, dtype=jnp.int32)

        init_state_free = [
            jnp.full((seq_len,), self.target_tokenizer.mask_id, dtype=jnp.int32)
            for _ in self.program_spec.gibbs_spec.free_blocks
        ]
        state_clamp = [encoded, mask_state]

        n_warmup = warmup if warmup is not None else int(self.schedule.n_levels - 1)
        schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=1, steps_per_sample=steps_per_sample)

        samples = sample_states(
            key,
            self.program_spec.program,
            schedule,
            init_state_free,
            state_clamp,
            [self.program_spec.latent_blocks[0]],
        )
        final_tokens = samples[0][0]
        return self.target_tokenizer.decode(final_tokens.tolist())


def _make_schedule(config: DiffusionTrainingConfig) -> MaskDiffusionSchedule:
    return default_mask_schedule(config.n_steps, min_keep=config.min_keep)


def _select_formats(curriculum: Sequence[tuple[int, List[int]]], epoch: int) -> Sequence[int] | None:
    for threshold, formats in sorted(curriculum, key=lambda x: x[0]):
        if epoch < threshold:
            return formats
    return None


def _build_training_assets(
    config: DiffusionTrainingConfig,
) -> tuple[
    DateExpressionGenerator,
    CharacterTokenizer,
    CharacterTokenizer,
    int,
    int,
]:
    generator = DateExpressionGenerator()
    input_tokenizer = build_default_tokenizer(config.max_length)
    target_tokenizer = build_iso_tokenizer(config.max_length)
    year_offset = generator.start_year
    year_classes = generator.end_year - generator.start_year + 1

    digit_token_ids = np.array([target_tokenizer.stoi[str(d)] for d in range(10)], dtype=np.int32)
    hyphen_token_id = int(target_tokenizer.stoi["-"])
    start_token_id = int(target_tokenizer.start_id)
    end_token_id = int(target_tokenizer.end_id)

    years = np.arange(generator.start_year, generator.end_year + 1)
    year_digits_table = np.array([[int(c) for c in f"{year:04d}"] for year in years], dtype=np.int32)
    month_digits_table = np.array([[int(c) for c in f"{month:02d}"] for month in range(1, 13)], dtype=np.int32)
    day_digits_table = np.array([[int(c) for c in f"{day:02d}"] for day in range(1, 32)], dtype=np.int32)

    iso_length = 10

    return (
        generator,
        input_tokenizer,
        target_tokenizer,
        year_offset,
        year_classes,
        digit_token_ids,
        hyphen_token_id,
        start_token_id,
        end_token_id,
        year_digits_table,
        month_digits_table,
        day_digits_table,
        iso_length,
    )


def _train_step_factory(
    schedule: MaskDiffusionSchedule,
    learning_rate: float,
    target_pad_id: int,
    target_mask_id: int,
    target_vocab_size: int,
    auxiliary_weight: float,
):
    optimizer = optax.adam(learning_rate)
    def diffuse_sample(key, clean_tokens, step):
        return diffuse_tokens_pair(
            key,
            clean_tokens,
            schedule=schedule,
            t=step,
            pad_id=target_pad_id,
            mask_id=target_mask_id,
        )

    def _cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        gathered = jnp.take_along_axis(log_probs, labels[:, None], axis=-1)
        return -jnp.mean(gathered)

    @eqx.filter_jit
    def train_step(
        model: TextDiffusionModel,
        opt_state,
        key: Array,
        inputs: Array,
        targets: Array,
        month_labels: Array,
        day_labels: Array,
        year_labels: Array,
    ):
        batch_size = targets.shape[0]

        def loss_fn(model, loss_key):
            step_key, noise_key = jax.random.split(loss_key)
            steps = jax.random.randint(step_key, (batch_size,), 0, schedule.n_levels - 1)
            noise_keys = jax.random.split(noise_key, batch_size)
            x_t, x_next = jax.vmap(diffuse_sample)(noise_keys, targets, steps)
            logits = jax.vmap(lambda prev, inp, step: model.compute_logits(prev, inp, step))(x_next, inputs, steps)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            clean_targets = targets
            token_log_probs = jnp.take_along_axis(log_probs, clean_targets[..., None], axis=-1)[..., 0]
            mask = (clean_targets != target_pad_id).astype(jnp.float32)
            normalizer = jnp.maximum(jnp.sum(mask), 1.0)
            main_loss = -jnp.sum(mask * token_log_probs) / normalizer

            month_logits, day_logits, year_logits = jax.vmap(model.component_logits)(inputs)
            month_loss = _cross_entropy(month_logits, month_labels)
            day_loss = _cross_entropy(day_logits, day_labels)
            year_loss = _cross_entropy(year_logits, year_labels)

            loss = main_loss + auxiliary_weight * (month_loss + day_loss + year_loss)
            return loss, (logits, mask)

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, key)
        logits, mask = aux
        predictions = jnp.argmax(logits, axis=-1)
        correct = jnp.sum(mask * (predictions == targets).astype(jnp.float32))
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        accuracy = correct / denom
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, accuracy

    return train_step, optimizer


def train_date_diffusion(config: DiffusionTrainingConfig) -> tuple[DateDiffusionPipeline, TrainingMetrics]:
    (
        generator,
        input_tokenizer,
        target_tokenizer,
        year_offset,
        year_classes,
        digit_token_ids,
        hyphen_token_id,
        start_token_id,
        end_token_id,
        year_digits_table,
        month_digits_table,
        day_digits_table,
        iso_length,
    ) = _build_training_assets(config)
    schedule = _make_schedule(config)

    key = jax.random.PRNGKey(config.seed)
    model = TextDiffusionModel.init(
        key,
        input_vocab_size=input_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
        hidden_dim=config.hidden_dim,
        n_steps=config.n_steps,
        target_mask_id=target_tokenizer.mask_id,
        target_pad_id=target_tokenizer.pad_id,
        input_pad_id=input_tokenizer.pad_id,
        sequence_length=config.max_length,
        year_offset=year_offset,
        year_classes=year_classes,
        digit_token_ids=digit_token_ids,
        hyphen_token_id=hyphen_token_id,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        year_digits_table=year_digits_table,
        month_digits_table=month_digits_table,
        day_digits_table=day_digits_table,
        iso_length=iso_length,
        encoder_layers=config.encoder_layers,
    )

    train_step, optimizer = _train_step_factory(
        schedule,
        config.learning_rate,
        target_pad_id=target_tokenizer.pad_id,
        target_mask_id=target_tokenizer.mask_id,
        target_vocab_size=target_tokenizer.vocab_size,
        auxiliary_weight=config.auxiliary_weight,
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    rng = random.Random(config.seed)
    losses: List[float] = []
    accuracies: List[float] = []

    for epoch in range(config.n_epochs):
        allowed_formats = _select_formats(config.curriculum_schedule, epoch)
        epoch_seed = rng.randint(0, 1_000_000_000)
        data = generate_date_dataset(
            config.dataset_size,
            input_tokenizer=input_tokenizer,
            output_tokenizer=target_tokenizer,
            seed=epoch_seed,
            generator=generator,
            allowed_formats=allowed_formats,
        )

        inputs = jnp.array(data["input_tokens"], dtype=jnp.int32)
        targets = jnp.array(data["target_tokens"], dtype=jnp.int32)
        month_labels = jnp.array(data["month_labels"], dtype=jnp.int32)
        day_labels = jnp.array(data["day_labels"], dtype=jnp.int32)
        year_labels = jnp.array(data["year_labels"], dtype=jnp.int32)

        dataset_size = targets.shape[0]
        batch_size = config.batch_size
        steps_per_epoch = max(math.ceil(dataset_size / batch_size), 1)

        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, dataset_size)
        inputs = inputs[perm]
        targets = targets[perm]
        month_labels = month_labels[perm]
        day_labels = day_labels[perm]
        year_labels = year_labels[perm]

        epoch_loss = 0.0
        epoch_acc = 0.0
        steps_run = 0

        for step in range(steps_per_epoch):
            start = step * batch_size
            if start >= dataset_size:
                break
            end = min(start + batch_size, dataset_size)
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            batch_month = month_labels[start:end]
            batch_day = day_labels[start:end]
            batch_year = year_labels[start:end]
            key, step_key = jax.random.split(key)
            model, opt_state, loss, acc = train_step(
                model,
                opt_state,
                step_key,
                batch_inputs,
                batch_targets,
                batch_month,
                batch_day,
                batch_year,
            )
            epoch_loss += float(loss)
            epoch_acc += float(acc)
            steps_run += 1

        denom = max(steps_run, 1)
        losses.append(epoch_loss / denom)
        accuracies.append(epoch_acc / denom)

    program_spec = build_text_diffusion_program(model, config.max_length)
    pipeline = DateDiffusionPipeline(
        input_tokenizer=input_tokenizer,
        target_tokenizer=target_tokenizer,
        model=model,
        schedule=schedule,
        program_spec=program_spec,
    )
    metrics = TrainingMetrics(epoch_losses=losses, epoch_accuracies=accuracies)
    return pipeline, metrics


def evaluate_pipeline(
    pipeline: DateDiffusionPipeline,
    *,
    n_samples: int = 256,
    warmup: int | None = None,
    steps_per_sample: int = 1,
    seed: int = 0,
) -> dict[str, float | list[tuple[str, str, str]] | list[float]]:
    generator = DateExpressionGenerator()
    rng = random.Random(seed)
    key = jax.random.PRNGKey(seed)
    warmup_steps = warmup if warmup is not None else pipeline.schedule.n_levels

    char_correct = 0
    char_total = 0
    sequence_correct = 0
    examples: list[tuple[str, str, str]] = []

    n_formats = generator.n_formatters
    char_correct_by_format = [0 for _ in range(n_formats)]
    char_total_by_format = [0 for _ in range(n_formats)]
    seq_correct_by_format = [0 for _ in range(n_formats)]
    seq_total_by_format = [0 for _ in range(n_formats)]

    for _ in range(n_samples):
        text, iso, fmt_idx, _ = generator.sample(rng)
        key, subkey = jax.random.split(key)
        prediction = pipeline.sample(text, key=subkey, warmup=warmup_steps, steps_per_sample=steps_per_sample)

        iso_str = iso.strip()
        pred_str = prediction.strip()
        length = len(iso_str)
        char_total += length
        char_correct += sum(1 for a, b in zip(pred_str[:length], iso_str) if a == b)

        char_total_by_format[fmt_idx] += length
        char_correct_by_format[fmt_idx] += sum(1 for a, b in zip(pred_str[:length], iso_str) if a == b)

        if pred_str[:length] == iso_str:
            sequence_correct += 1
            seq_correct_by_format[fmt_idx] += 1
        seq_total_by_format[fmt_idx] += 1

        if len(examples) < 5:
            examples.append((text, iso_str, pred_str))

    per_format_char = [
        (char_correct_by_format[i] / char_total_by_format[i]) if char_total_by_format[i] else 0.0
        for i in range(n_formats)
    ]
    per_format_seq = [
        (seq_correct_by_format[i] / seq_total_by_format[i]) if seq_total_by_format[i] else 0.0
        for i in range(n_formats)
    ]

    return {
        "char_accuracy": char_correct / max(char_total, 1),
        "sequence_accuracy": sequence_correct / n_samples,
        "examples": examples,
        "per_format_char_accuracy": per_format_char,
        "per_format_sequence_accuracy": per_format_seq,
    }
