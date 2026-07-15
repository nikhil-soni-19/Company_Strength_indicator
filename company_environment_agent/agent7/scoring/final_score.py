def combine(quant: float, qual: float) -> dict:
    score = round(0.5 * quant + 0.5 * qual, 2)
    if   score >= 70: direction = "SUPPORTIVE"
    elif score >= 30: direction = "MIXED"
    else:             direction = "HOSTILE"
    return {"environment_score": score, "direction": direction}
