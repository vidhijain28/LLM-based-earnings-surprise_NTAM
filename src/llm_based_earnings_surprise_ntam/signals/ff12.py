"""Fama-French 12 industries from SIC codes.

SIC ranges follow Kenneth French's 12-industry definition, taken from a public copy of the definition file.
Industry 12 (Other) is every SIC code that is not in industries 1 to 11.

    from llm_based_earnings_surprise_ntam.signals.ff12 import ff12_code, ff12_name
    df["industry"] = ff12_name(df["sic"])        # a Series in, a Series out (missing or invalid SIC gives NaN)
    ff12_name(3571)                              # "BusEq"

The pipeline's final file has no SIC column. Compustat's `sic` (from `comp.company`, or a Compustat export) is the
value at the download date, not the value on each event date. That is fine for grouping events, but it is not point-in-time.

Self-test: python -m llm_based_earnings_surprise_ntam.signals.ff12
"""
import numpy as np
import pandas as pd

FF12_NAMES = {1: "NoDur", 2: "Durbl", 3: "Manuf", 4: "Enrgy", 5: "Chems", 6: "BusEq",
              7: "Telcm", 8: "Utils", 9: "Shops", 10: "Hlth", 11: "Money", 12: "Other"}

_RANGES = {
    1:  [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    2:  [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    3:  [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)],
    4:  [(1200, 1399), (2900, 2999)],
    5:  [(2800, 2829), (2840, 2899)],
    6:  [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    7:  [(4800, 4899)],
    8:  [(4900, 4949)],
    9:  [(5000, 5999), (7200, 7299), (7600, 7699)],
    10: [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    11: [(6000, 6999)],
}

# Lookup over all four-digit SIC codes; 12 (Other) is the default.
_LOOKUP = np.full(10000, 12, dtype=np.int8)
_HITS = np.zeros(10000, dtype=np.int8)
for _code, _ranges in _RANGES.items():
    for _a, _b in _ranges:
        _LOOKUP[_a:_b + 1] = _code
        _HITS[_a:_b + 1] += 1
assert _HITS.max() <= 1, "a SIC code falls into more than one industry"


def ff12_code(sic):
    """Industry code 1-12. A scalar gives an int (None if missing or invalid); a Series or array gives a float Series (NaN if missing)."""
    if np.isscalar(sic) or sic is None:
        v = pd.to_numeric(pd.Series([sic]), errors="coerce").iloc[0]
        return None if pd.isna(v) or not 0 <= v <= 9999 else int(_LOOKUP[int(v)])
    s = pd.to_numeric(pd.Series(sic), errors="coerce")
    ok = s.notna() & (s >= 0) & (s <= 9999)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[ok] = _LOOKUP[s[ok].astype(int).to_numpy()]
    return out


def ff12_name(sic):
    """Industry name (see FF12_NAMES), same input and output rules as ff12_code."""
    code = ff12_code(sic)
    if isinstance(code, pd.Series):
        return code.map(FF12_NAMES)
    return None if code is None else FF12_NAMES[code]


def _selftest():
    spot = {3571: "BusEq", 7372: "BusEq", 2834: "Hlth", 3841: "Hlth", 6021: "Money", 6311: "Money", 1311: "Enrgy", 2911: "Enrgy",
            4911: "Utils", 4813: "Telcm", 5812: "Shops", 5411: "Shops", 2000: "NoDur", 2111: "NoDur", 3711: "Durbl", 3490: "Manuf",
            2820: "Chems", 7011: "Other", 1000: "Other", 8742: "Other"}
    for sic, name in spot.items():
        assert ff12_name(sic) == name, (sic, ff12_name(sic), name)
    assert ff12_name("3571") == "BusEq" and ff12_name(float("nan")) is None and ff12_name(None) is None and ff12_name(-5) is None
    s = pd.Series([3571.0, np.nan, 6021.0, 99999.0], index=[10, 11, 12, 13])
    out = ff12_name(s)
    assert list(out.index) == [10, 11, 12, 13] and out[10] == "BusEq" and pd.isna(out[11]) and out[12] == "Money" and pd.isna(out[13])
    assert set(FF12_NAMES) == set(range(1, 13)) and int(_HITS.max()) == 1
    print("ff12 self-test passed:", len(spot), "spot checks, no overlapping ranges")


if __name__ == "__main__":
    _selftest()
