
import pandas as pd

def format_currency(value):
    return f"Rp {value:,.0f}".replace(",", ".")
