IDEAL_MCS = True
SPATIAL_REUSE = True


def timestamp(time: float) -> str:
    t = f"{10**6 * time:.9f} us"
    leading_zeros = max(6 - len(t.split(".")[0]), 0)
    return "0" * leading_zeros + t
