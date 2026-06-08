"""
.dcupt (delta cupt) file format.

A .dcupt file describes a variant of a base .cupt file by specifying:
- Which sentences to include (sentence selection)
- Column overrides (typically the parseme:mwe column)

This avoids duplicating entire .cupt files when only the MWE annotations
or sentence selection differ.

Format specification
--------------------

Single-base example::

    # dcupt-version = 1.1
    # base = relative/path/to/base.cupt
    # columns = PARSEME:MWE                   (columns to override)
    # default-value = *                       (blanket value for all tokens)

    # sentence = 3
    *
    1:VPC.full
    *
    *

    # sentence = 10
    *
    *

Multi-base example (two bases, each with their own sentence counter)::

    # dcupt-version = 1.1
    # base = train.blind.cupt
    # base = dev.blind.cupt
    # columns = PARSEME:MWE
    # default-value = *

    # sentence = 0:3
    *
    1:VPC.full
    *

    # sentence = 1:2
    *
    2:IRV
    *

Sentence inclusion rules
~~~~~~~~~~~~~~~~~~~~~~~~

For each base independently:

- If **any** ``# sentence = B:S`` block references that base → only the listed
  sentences from that base are included in the output.
- If **no** block references a base → **all** sentences from that base are
  included (pass-through).

This means sentence selection and column overrides share the same mechanism:
a block with no token lines means "include this sentence, apply default-value
to all tokens".

Notes:
    - base must be a regular .cupt file (no recursive .dcupt references)
    - base path is relative to the .dcupt file location
    - sentence indices are 0-based positions within their respective base file
    - single-base files use ``# sentence = S``; multi-base files use
      ``# sentence = B:S`` where B is the 0-based base index
    - plain ``# sentence = S`` is always read as ``(0, S)`` for backward compat
    - per-sentence overrides take precedence over default-value
    - override values correspond 1:1 to token lines in the base sentence
    - for multiple override columns, values are tab-separated
"""

import io
from pathlib import Path
from conllu import parse_incr, TokenList
from typing import Iterator, Optional, Union
import re

DCUPT_EXTENSION = '.dcupt'


def _parse_sentence_key(value: str) -> tuple[int, int]:
    """Parse a sentence key in ``B:S`` or ``S`` format to a (base, sent) tuple."""
    if ':' in value:
        b, s = value.split(':', 1)
        return (int(b), int(s))
    return (0, int(value))


def is_dcupt(path: str) -> bool:
    """Check if a file path refers to a .dcupt file."""
    return path.endswith(DCUPT_EXTENSION)


def _parse_file(filepath: str) -> tuple[dict[str, Union[str, list[str]]], dict[tuple[int, int], list[str]]]:
    """Parse a .dcupt file into header and per-sentence overrides.

    Returns
    -------
    header : dict[str, str | list[str]]
        Key-value pairs from the header. 'base' may be a list[str] when
        multiple ``# base = ...`` lines are present.
    overrides : dict[tuple[int, int], list[str]]
        Per-sentence column override values keyed by ``(base_index, sentence_index)``.
        ``# sentence = S`` is read as ``(0, S)`` for backward compatibility.
        Each value is a list of strings, one per token line.
    """
    header: dict[str, Union[str, list[str]]] = {}
    overrides: dict[tuple[int, int], list[str]] = {}
    current_sentence: Optional[tuple[int, int]] = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            stripped = line.strip()

            if not stripped:
                current_sentence = None
                continue

            if stripped.startswith('#'):
                m = re.match(r'^#\s*([\w-]+)\s*=\s*(.+)$', stripped)
                if m:
                    key, value = m.group(1), m.group(2).strip()
                    if key == 'sentence':
                        current_sentence = _parse_sentence_key(value)
                        overrides[current_sentence] = []
                    elif key == 'base':
                        if 'base' not in header:
                            header['base'] = value
                        else:
                            existing = header['base']
                            if isinstance(existing, list):
                                existing.append(value)
                            else:
                                header['base'] = [existing, value]
                    else:
                        header[key] = value
                continue

            if current_sentence is not None:
                overrides[current_sentence].append(stripped)

    return header, overrides


def _validate_and_resolve_bases(header: dict[str, Union[str, list[str]]], filepath: str) -> list[Path]:
    """Validate dcupt header and return list of resolved base file paths."""
    version = header.get('dcupt-version', '1.1')
    if version not in ('1', '1.1'):
        raise ValueError(f"Unsupported dcupt version: {version}")

    if 'base' not in header:
        raise ValueError(f"Missing required 'base' header in: {filepath}")

    base_val = header['base']
    base_refs = base_val if isinstance(base_val, list) else [base_val]

    dcupt_dir = Path(filepath).resolve().parent
    base_paths: list[Path] = []
    for ref in base_refs:
        base_path = (dcupt_dir / ref.replace('\\', '/')).resolve()
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base file not found: {base_path} "
                f"(referenced from {filepath})"
            )
        if is_dcupt(str(base_path)):
            raise ValueError(
                f"Recursive .dcupt references are not supported. "
                f"Base must be a .cupt file: {base_path}"
            )
        base_paths.append(base_path)
    return base_paths


def resolve_as_stream(filepath: str) -> io.StringIO:
    """Resolve a .dcupt file into a text stream indistinguishable from a .cupt file.

    Reads the base .cupt file(s) verbatim, applies column replacements and
    sentence filtering at the text level. The output is byte-for-byte
    identical to a real .cupt file (modulo the overridden columns).

    Sentence inclusion is per-base: if any block references a base, only the
    listed sentences from that base are included; otherwise all are included.
    """
    header, overrides = _parse_file(filepath)
    base_paths = _validate_and_resolve_bases(header, filepath)

    # Per-base sentence selection derived from override keys.
    # base_selections[b] = set of local sentence indices to include from base b.
    # If base b has no entry → include all sentences from that base.
    base_selections: dict[int, set[int]] = {}
    for (b, s) in overrides:
        base_selections.setdefault(b, set()).add(s)

    # Column names to override
    columns = header['columns'].split() if 'columns' in header else []
    default_value = header.get('default-value')

    out_lines: list[str] = []

    for base_idx, base_path in enumerate(base_paths):
        # Find column indices from # global.columns header line in this base
        col_indices: dict[str, int] = {}
        with open(str(base_path), 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(r'^#\s*global\.columns\s*=\s*(.+)$', line, re.IGNORECASE)
                if m:
                    col_names = m.group(1).strip().split()
                    for col in columns:
                        for i, cn in enumerate(col_names):
                            if cn.upper() == col.upper():
                                col_indices[col] = i
                                break
                    break

        # None means "include all sentences from this base"
        base_selected = base_selections.get(base_idx)

        # Stream through the base file line by line
        local_sent_idx = 0  # sentence index within this base
        token_idx = 0
        in_sentence = False
        has_data = False  # did current block have any data (non-comment) lines?

        with open(str(base_path), 'r', encoding='utf-8') as f:
            for line in f:
                raw = line.rstrip('\n\r')

                # Empty line = sentence boundary
                if raw.strip() == '':
                    if in_sentence and has_data:
                        local_sent_idx += 1
                        token_idx = 0
                        out_lines.append('')
                    # If the block had no data lines (e.g. standalone
                    # # global.columns), skip the blank line so it merges
                    # with the next block.
                    in_sentence = False
                    has_data = False
                    continue

                in_sentence = True

                # Comment lines: pass through unchanged
                if raw.startswith('#'):
                    if has_data and base_selected is not None and local_sent_idx not in base_selected:
                        continue
                    out_lines.append(raw)
                    continue

                # First data line in block — now we know it's a real sentence
                if not has_data:
                    has_data = True
                    if base_selected is not None and local_sent_idx not in base_selected:
                        while out_lines and out_lines[-1].startswith('#'):
                            out_lines.pop()
                        continue

                # Skip data lines for filtered sentences
                if base_selected is not None and local_sent_idx not in base_selected:
                    continue

                # Data line: apply column overrides
                if col_indices:
                    parts = raw.split('\t')
                    sent_ov = overrides.get((base_idx, local_sent_idx))
                    if sent_ov is not None and token_idx < len(sent_ov):
                        ov_parts = sent_ov[token_idx].split('\t')
                        for j, col in enumerate(columns):
                            if col in col_indices and j < len(ov_parts):
                                parts[col_indices[col]] = ov_parts[j]
                    elif default_value is not None:
                        for col in columns:
                            if col in col_indices:
                                parts[col_indices[col]] = default_value
                    out_lines.append('\t'.join(parts))
                else:
                    out_lines.append(raw)
                token_idx += 1

    return io.StringIO('\n'.join(out_lines) + '\n')


def resolve(filepath: str) -> Iterator[TokenList]:
    """Resolve a .dcupt file into an iterator of TokenLists."""
    return parse_incr(resolve_as_stream(filepath))


def create(
    output_path: str,
    base_ref: Union[str, list[str]],
    columns: Optional[list[str]] = None,
    default_value: Optional[str] = None,
    overrides: Optional[dict[tuple[int, int], list[str]]] = None,
) -> None:
    """Create a .dcupt file.

    Parameters
    ----------
    output_path : str
        Where to write the .dcupt file.
    base_ref : str | list[str]
        Path(s) to the base .cupt file(s), stored as-is in the header.
        Should be relative to the .dcupt file location.
    columns : list[str] | None
        Column names to override (e.g. ['PARSEME:MWE']).
    default_value : str | None
        Blanket default value for overridden columns.
    overrides : dict[tuple[int, int], list[str]] | None
        Per-sentence data keyed by ``(base_index, sentence_index)``.

        - A non-empty list provides per-token replacement values for the
          overridden columns.
        - An empty list ``[]`` means: include this sentence, apply
          ``default_value`` to all its tokens (no explicit per-token data).

        If a base has at least one entry in *overrides*, only the listed
        sentences are included from that base.  Bases with no entries are
        included in full (pass-through).
    """
    lines = ['# dcupt-version = 1.1']
    base_refs = base_ref if isinstance(base_ref, list) else [base_ref]
    multi = len(base_refs) > 1
    for ref in base_refs:
        lines.append(f'# base = {ref}')

    def _fmt_key(b: int, s: int) -> str:
        return f'{b}:{s}' if multi else str(s)

    if columns:
        lines.append(f'# columns = {" ".join(columns)}')

    if default_value is not None:
        lines.append(f'# default-value = {default_value}')

    lines.append('')

    if overrides:
        for (b, s) in sorted(overrides):
            lines.append(f'# sentence = {_fmt_key(b, s)}')
            lines.extend(overrides[(b, s)])
            lines.append('')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def count_sentences(filepath: str) -> int:
    """Count the number of sentences in a .cupt file."""
    count = 0
    has_data = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                has_data = True
            elif not stripped and has_data:
                count += 1
                has_data = False
    if has_data:
        count += 1
    return count


def read_overrides(filepath: str) -> dict[tuple[int, int], list[str]]:
    """Return the per-sentence overrides stored in a .dcupt file.

    The returned dict maps ``(base_index, sentence_index)`` tuples to lists
    of per-token override strings.
    """
    _, overrides = _parse_file(filepath)
    return overrides
