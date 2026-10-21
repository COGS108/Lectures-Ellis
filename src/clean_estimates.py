"""
Clean free-text Fermi-estimate responses into numbers so the median can be computed.

Usage:
    df = pd.read_csv('responses.tsv', sep='\t')
    df.columns = ['hair_cm_yr', 'area_km2', 'walk_days']
    df['hair_cm_yr'] = df['hair_cm_yr'].apply(clean_hair)
    df['area_km2']   = df['area_km2'].apply(clean_area)
    df['walk_days']  = df['walk_days'].apply(clean_days)
    df.median()

Anything that can't be confidently turned into a number becomes NaN,
which pandas skips when computing the median.
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# One "number" in a response. The forms are tried in order, so
# 6x10^7 is read as one number, not as 6, 10, and 7.
NUMBER = re.compile(r"""
    (?P<num>
        \d+(?:\.\d+)?\s*[x*]\s*10\s*\^\s*-?\d+     # 6x10^7, 1 x 10^2
      | 10\s*\^\s*-?\d+                            # 10^8
      | \d+(?:\.\d+)?e[+-]?\d+                     # 1e+15
      | \d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?          # 1/4, 10/1 (fractions)
      | \d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?          # 12-15 (ranges -> midpoint)
      | \d+(?:\.\d+)?                              # 42, 8.78
    )
    (?:\s*(?P<mult>thousand|million|mil|billion|bil|trillion|hundred|k)\b)?
    (?:\s*(?P<unit>[a-z]+))?                        # word right after the number
""", re.VERBOSE)

MULTIPLIERS = {'hundred': 1e2, 'thousand': 1e3, 'k': 1e3,
               'million': 1e6, 'mil': 1e6,
               'billion': 1e9, 'bil': 1e9, 'trillion': 1e12}

WORD_NUMBERS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}


def _to_float(token):
    """Convert one matched number token (any form above) to a float."""
    token = token.replace(' ', '')
    if '^' in token:
        base, exp = token.split('^')
        coef = re.split(r'[x*]', base)[0] if re.search(r'[x*]', base) else '1'
        return float(coef) * 10.0 ** float(exp)
    if '/' in token:
        num, den = token.split('/')
        return float(num) / float(den) if float(den) != 0 else np.nan
    if re.fullmatch(r'[\d.]+-[\d.]+', token):
        low, high = token.split('-')
        return (float(low) + float(high)) / 2
    return float(token)


def _standardize(value):
    """Lowercase, trim, and remove thousands separators (1,000,000 -> 1000000)."""
    if not isinstance(value, str):
        return None
    text = value.lower().strip()
    text = re.sub(r'(?<=\d),(?=\d)', '', text)   # commas between digits
    text = text.replace('×', 'x')
    text = re.sub(r'/\s*1\s*(?=y)', '/', text)   # '50cm/1yr' -> '50cm/yr'
    return text


def _extract(text, unit_factor, target_unit):
    """
    Pull a single number out of `text` and convert it to the target unit.

    unit_factor(word) -> multiplier to reach the target unit, or None if the
                         word isn't a valid unit for this question.
    target_unit       -> prefix of the unit asked for (e.g. 'cm'). If a response
                         has several numbers, we keep the one labeled with
                         this unit; if that's ambiguous, we return NaN.
    """
    if not text:
        return np.nan

    candidates = []
    for m in NUMBER.finditer(text):
        value = _to_float(m.group('num'))
        if m.group('mult'):
            value *= MULTIPLIERS[m.group('mult')]
        candidates.append((value, m.group('unit')))

    if len(candidates) == 0:
        return np.nan
    if len(candidates) > 1:
        candidates = [c for c in candidates
                      if c[1] is not None and c[1].startswith(target_unit)]
        if len(candidates) != 1:
            return np.nan

    value, unit = candidates[0]
    factor = unit_factor(unit)
    return np.nan if factor is None else value * factor


# ---------------------------------------------------------------------------
# Column-specific cleaners
# ---------------------------------------------------------------------------

def clean_hair(value):
    """Hair growth -> cm/yr. Bare numbers are assumed to be cm; inches converted."""

    def unit_factor(word):
        if word is None or word.startswith('c'):     # cm, cm/yr
            return 1
        if word.startswith('in'):                    # inch, inches
            return 2.54
        if word.startswith('mm'):
            return 0.1
        return None                                  # unrecognized unit

    return _extract(_standardize(value), unit_factor, target_unit='cm')


def clean_area(value):
    """Area -> km². Bare numbers and plain 'km' are assumed to mean km²."""
    text = _standardize(value)
    if text is None or 'few' in text:                # "a few billion..." is too vague
        return np.nan

    # Collapse all the ways of writing km² to 'km' so the 2 isn't read as a number
    text = re.sub(r'(km|kilomet\w*)\s*(\^\s*\d+|²|2|\s+squared?|\s+sq\w*)?', ' km ', text)
    text = re.sub(r'square\s+km', 'km', text)
    # "a million" -> "1 million"
    text = re.sub(r'\ban?\s+(?=(hundred|thousand|million|billion|trillion)\b)', '1 ', text)

    def unit_factor(word):
        if word is None or word.startswith('km'):
            return 1
        return None                                  # miles, ft, etc. aren't areas

    return _extract(text, unit_factor, target_unit='km')


def clean_days(value):
    """Walking time -> days. Bare numbers are assumed to be days."""
    text = _standardize(value)
    if text is None:
        return np.nan

    text = re.sub(r'(\d+)\s+and\s+a\s+half', r'\1.5', text)        # 1 and a half
    # "1 year 6 months" -> "1.5 years"
    text = re.sub(r'(\d+(?:\.\d+)?)\s*y\w*\s+(?:and\s+)?(\d+(?:\.\d+)?)\s*mo\w*',
                  lambda m: f"{float(m.group(1)) + float(m.group(2)) / 12} years", text)
    for word, num in WORD_NUMBERS.items():                          # two weeks
        text = re.sub(rf'\b{word}\b', str(num), text)
    # "a week", "an entire year" -> "1 week", "1 year"
    text = re.sub(r'\ban?\s+(?:entire\s+|whole\s+)?(?=(day|week|month|year))', '1 ', text)

    def unit_factor(word):
        if word is None or word.startswith('d'):    # days (and typos like 'dsys')
            return 1
        if word.startswith('h'):                    # hours
            return 1 / 24
        if word.startswith('w'):                    # weeks
            return 7
        if word.startswith('mo'):                   # months
            return 30
        if word.startswith('y'):                    # years, yrs
            return 365
        return None                                 # km, miles, mph...

    return _extract(text, unit_factor, target_unit='d')
