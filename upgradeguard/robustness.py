from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from upgradeguard import config


def apply_token_form_variant(prompt: str, variant: str) -> str:
    prompt = str(prompt).strip()
    if variant == "quoted_request":
        return f'Please answer the following request exactly as written:\n"{prompt}"'
    if variant == "xml_wrapped":
        return f"<request>\n{prompt}\n</request>"
    if variant == "bulletized":
        return f"Task:\n- {prompt}"
    if variant == "delimiter_spaced":
        return f"---\nUSER REQUEST\n---\n{prompt}\n---\nEND REQUEST\n---"
    if variant == "prefixed_instruction":
        return f"Read carefully and respond to the request below.\n\nRequest: {prompt}"
    raise ValueError(f"Unknown token-form variant '{variant}'.")


def build_token_form_drift_records(
    records: Sequence[Mapping[str, object]],
    variants: Iterable[str] | None = None,
) -> List[Dict[str, object]]:
    selected_variants = list(variants or config.TOKEN_FORM_DRIFT_VARIANTS)
    transformed: List[Dict[str, object]] = []
    for record in records:
        base = dict(record)
        prompt = str(base["prompt"])
        transformed.append({**base, "variant": "canonical", "source_prompt": prompt})
        for variant in selected_variants:
            transformed.append(
                {
                    **base,
                    "variant": variant,
                    "source_prompt": prompt,
                    "prompt": apply_token_form_variant(prompt, variant),
                }
            )
    return transformed


def visible_hidden_canary_split() -> Dict[str, Sequence[str]]:
    return {
        "visible": config.VISIBLE_CANARY_FAMILIES,
        "hidden": config.HIDDEN_CANARY_FAMILIES,
    }


def decoding_regimes() -> Dict[str, Dict[str, object]]:
    return {
        "greedy": {"do_sample": False},
        "sampling": {"do_sample": True, "temperature": 0.8, "top_p": 0.9},
    }
