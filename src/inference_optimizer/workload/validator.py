"""Dataset validation policies, stated explicitly:

1. IDs must be unique across the dataset (a duplicate id makes sampling/lookup
   ambiguous).
2. Prompt text must be non-empty.
3. `input_tokens_measured` must be > 0 and must never exceed `context_limit -
   output_target_tokens - safety_margin` for that record -- this is the "stays
   within model limits" invariant from the brief, checked against the actual
   measured count, not the label.
4. `input_tokens_measured` should be within `tolerance` of the record's
   *effective* target (its `input_length_target` label, clamped the same way the
   generator clamps the 2048 tier -- see generator.py) -- a WARNING, not an
   error: a user-provided dataset may legitimately miss the target length
   (e.g. a real document that doesn't tokenize to exactly 512), so this is
   reported but does not fail validation on its own.

`validate_dataset` returns (errors, warnings) -- both lists of human-readable
strings. An empty `errors` list means the dataset is structurally usable; a
non-empty `warnings` list flags things worth a human's attention.
"""

from __future__ import annotations

from inference_optimizer.workload.models import PromptRecord

_DEFAULT_TOLERANCE = 0.25
_DEFAULT_SAFETY_MARGIN = 32


def _effective_target(record: PromptRecord, context_limit: int, safety_margin: int) -> int:
    max_input_for_context = context_limit - record.output_target_tokens - safety_margin
    return min(record.input_length_target, max(max_input_for_context, 0))


def validate_dataset(
    records: list[PromptRecord],
    context_limit: int = 2048,
    tolerance: float = _DEFAULT_TOLERANCE,
    safety_margin: int = _DEFAULT_SAFETY_MARGIN,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not records:
        errors.append("Dataset is empty.")
        return errors, warnings

    seen_ids: set[str] = set()
    for record in records:
        if record.id in seen_ids:
            errors.append(f"Duplicate id: {record.id!r}.")
        seen_ids.add(record.id)

        if not record.prompt.strip():
            errors.append(f"{record.id}: prompt is empty.")

        if record.input_tokens_measured <= 0:
            errors.append(f"{record.id}: input_tokens_measured must be > 0.")

        max_allowed = context_limit - record.output_target_tokens - safety_margin
        if record.input_tokens_measured > max_allowed:
            errors.append(
                f"{record.id}: input_tokens_measured={record.input_tokens_measured} exceeds "
                f"the model-limit budget ({max_allowed} = context_limit {context_limit} - "
                f"output_target {record.output_target_tokens} - safety_margin {safety_margin})."
            )

        effective_target = _effective_target(record, context_limit, safety_margin)
        if effective_target > 0:
            lower = effective_target * (1 - tolerance)
            upper = effective_target * (1 + tolerance)
            if not (lower <= record.input_tokens_measured <= upper):
                warnings.append(
                    f"{record.id}: input_tokens_measured={record.input_tokens_measured} is "
                    f"outside +/-{tolerance:.0%} of its effective target ({effective_target})."
                )

    return errors, warnings
