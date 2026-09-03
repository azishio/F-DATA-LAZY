"""Anonymization of jid/usr/jnam/jobenv_req via salted-hash labels.

Labels are keyed hashes: `usr_<64 hex chars of HMAC-SHA256(salt, value)>`.
Deterministic given the salt, so outputs of separate runs sharing a salt can
be combined (same original -> same label). The salt is what prevents
dictionary attacks on low-entropy identifiers — an unsalted hash of a
guessable username is trivially reversible — so it must be kept secret and
reused across runs that need to interoperate.

The mapping tables (unique users/job names) are tiny compared to the data:
one streaming pass collects the distinct values and the labels are applied
via small in-memory left joins.

Divergences from the original scripts (documented in the README): the
published F-DATA uses first-appearance sequential numbering ("usr_0",
"usr_1", ...) instead of hashes, and mapped NaN to a pseudonym where we
keep nulls null.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timezone

import polars as pl

from .schema import ANON_FEATURES


def generate_salt() -> str:
    """Auto-salt for runs that don't pass one: current time plus random
    bits (time alone would be guessable enough to enable dictionary
    attacks). Logged by the CLI so later runs can reuse it."""
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{now}-{os.urandom(8).hex()}"


def hash_label(feat: str, value: str, salt: str) -> str:
    digest = hmac.new(
        salt.encode(), f"{feat}:{value}".encode(), hashlib.sha256
    ).hexdigest()
    return f"{feat}_{digest}"


def build_maps(lf: pl.LazyFrame, salt: str) -> dict[str, pl.DataFrame]:
    if not salt:
        raise ValueError("anonymization requires a salt")
    plans = [
        lf.select(pl.col(feat).drop_nulls()).unique() for feat in ANON_FEATURES
    ]
    uniques = pl.collect_all(plans, engine="streaming")
    maps = {}
    for feat, frame in zip(ANON_FEATURES, uniques):
        values = frame.to_series().to_list()
        mapping = pl.DataFrame(
            {
                f"{feat}_or": pl.Series(values, dtype=pl.Utf8),
                feat: [hash_label(feat, v, salt) for v in values],
            }
        )
        anonymized_count = mapping[feat].n_unique()
        if anonymized_count != len(values):
            raise RuntimeError(
                f"Anonymization collision in {feat}: {len(values)} unique "
                f"values before, {anonymized_count} after"
            )
        maps[feat] = mapping
    return maps


def anonymize(lf: pl.LazyFrame, maps: dict[str, pl.DataFrame]) -> pl.LazyFrame:
    for feat, mapping in maps.items():
        lf = lf.rename({feat: f"{feat}_or"}).join(
            mapping.lazy(), on=f"{feat}_or", how="left", maintain_order="left"
        )
    return lf
