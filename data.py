from __future__ import annotations

import calendar
import dataclasses
import random
from collections.abc import Iterable, Sequence
from datetime import date, timedelta


@dataclasses.dataclass
class CharacterTokenizer:
    """Simple character-level tokenizer with padding and masking support."""

    max_length: int
    alphabet: Iterable[str]
    pad_token: str = "<pad>"
    mask_token: str = "<mask>"
    unk_token: str = "<unk>"
    start_token: str = "<s>"
    end_token: str = "</s>"

    def __post_init__(self) -> None:
        base_tokens = [
            self.pad_token,
            self.mask_token,
            self.unk_token,
            self.start_token,
            self.end_token,
        ]
        seen = set(base_tokens)
        filtered = [ch for ch in self.alphabet if ch not in seen]
        vocab = base_tokens + filtered
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.vocab_size = len(vocab)
        self.pad_id = self.stoi[self.pad_token]
        self.mask_id = self.stoi[self.mask_token]
        self.unk_id = self.stoi[self.unk_token]
        self.start_id = self.stoi[self.start_token]
        self.end_id = self.stoi[self.end_token]

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        text = text.lower()
        tokens = []
        if add_special_tokens:
            tokens.append(self.start_id)
        tokens.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        if add_special_tokens:
            tokens.append(self.end_id)
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens += [self.pad_id] * (self.max_length - len(tokens))
        return tokens

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        chars: list[str] = []
        special_ids = {self.pad_id, self.mask_id, self.start_id, self.end_id}
        if skip_special_tokens:
            special_ids.add(self.unk_id)
        for idx in token_ids:
            token = self.itos.get(int(idx), self.unk_token)
            if skip_special_tokens and self.stoi.get(token, -1) in special_ids:
                continue
            chars.append(token)
        return "".join(chars)


def _ordinal_suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


_DAY_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    21: "twenty first",
    22: "twenty second",
    23: "twenty third",
    24: "twenty fourth",
    25: "twenty fifth",
    26: "twenty sixth",
    27: "twenty seventh",
    28: "twenty eighth",
    29: "twenty ninth",
    30: "thirtieth",
    31: "thirty first",
}


@dataclasses.dataclass
class DateExpressionGenerator:
    """Sample textual date expressions paired with ISO targets."""

    start_year: int = 1900
    end_year: int = 2100

    def __post_init__(self) -> None:
        if self.end_year <= self.start_year:
            raise ValueError("end_year must exceed start_year")
        self._formatters = [
            self._month_name_day_year,
            self._weekday_month_day_year,
            self._ordinal_day_month_year,
            self._numeric_slash,
            self._numeric_dots,
            self._textual_clause,
        ]

    def sample(
        self,
        rng: random.Random | None = None,
        allowed_formats: Sequence[int] | None = None,
    ) -> tuple[str, str, int, date]:
        rng = rng or random
        sampled_date = self._random_date(rng)
        if allowed_formats:
            fmt_idx = rng.choice(list(allowed_formats))
        else:
            fmt_idx = rng.randrange(len(self._formatters))
        formatter = self._formatters[fmt_idx]
        return formatter(sampled_date), sampled_date.strftime("%Y-%m-%d"), fmt_idx, sampled_date

    @property
    def n_formatters(self) -> int:
        return len(self._formatters)

    def _random_date(self, rng: random.Random) -> date:
        start = date(self.start_year, 1, 1)
        end = date(self.end_year, 12, 31)
        delta_days = (end - start).days
        return start + timedelta(days=rng.randint(0, delta_days))

    def _month_name_day_year(self, d: date) -> str:
        month = calendar.month_name[d.month].lower()
        return f"{month} {d.day}, {d.year}"

    def _weekday_month_day_year(self, d: date) -> str:
        weekday = calendar.day_name[d.weekday()].lower()
        month = calendar.month_name[d.month].lower()
        suffix = _ordinal_suffix(d.day)
        return f"{weekday}, {month} {d.day}{suffix} {d.year}"

    def _ordinal_day_month_year(self, d: date) -> str:
        month = calendar.month_name[d.month].lower()
        suffix = _ordinal_suffix(d.day)
        return f"{d.day}{suffix} of {month} {d.year}"

    def _numeric_slash(self, d: date) -> str:
        return d.strftime("%m/%d/%Y")

    def _numeric_dots(self, d: date) -> str:
        return d.strftime("%d.%m.%Y")

    def _textual_clause(self, d: date) -> str:
        month = calendar.month_name[d.month].lower()
        day_word = _DAY_WORDS[d.day]
        return f"the {day_word} day of {month} in {d.year}"


def build_default_tokenizer(max_length: int) -> CharacterTokenizer:
    base_chars = "abcdefghijklmnopqrstuvwxyz0123456789 ,-/'\\.:"
    unique_chars = sorted(set(base_chars))
    return CharacterTokenizer(max_length=max_length, alphabet=unique_chars)


def build_iso_tokenizer(max_length: int) -> CharacterTokenizer:
    base_chars = "0123456789-"
    return CharacterTokenizer(max_length=max_length, alphabet=base_chars)


def generate_date_dataset(
    n_samples: int,
    *,
    input_tokenizer: CharacterTokenizer,
    output_tokenizer: CharacterTokenizer,
    seed: int | None = None,
    generator: DateExpressionGenerator | None = None,
    allowed_formats: Sequence[int] | None = None,
) -> dict[str, list[list[int]] | list[str] | list[int]]:
    rng = random.Random(seed)
    generator = generator or DateExpressionGenerator()
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    raw_inputs: list[str] = []
    raw_targets: list[str] = []
    month_labels: list[int] = []
    day_labels: list[int] = []
    year_labels: list[int] = []
    format_ids: list[int] = []
    for _ in range(n_samples):
        text, iso, fmt_idx, dt = generator.sample(rng, allowed_formats=allowed_formats)
        raw_inputs.append(text)
        raw_targets.append(iso)
        inputs.append(input_tokenizer.encode(text))
        targets.append(output_tokenizer.encode(iso))
        month_labels.append(dt.month - 1)
        day_labels.append(dt.day - 1)
        year_labels.append(dt.year - generator.start_year)
        format_ids.append(fmt_idx)
    return {
        "input_tokens": inputs,
        "target_tokens": targets,
        "raw_inputs": raw_inputs,
        "raw_targets": raw_targets,
        "month_labels": month_labels,
        "day_labels": day_labels,
        "year_labels": year_labels,
        "format_ids": format_ids,
    }
