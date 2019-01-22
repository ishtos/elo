# =============================================================================
# 000 initializer
# =============================================================================
python -u 001_init.py

# =============================================================================
# 100 historical
# =============================================================================
python -u 101_init.py
python -u 102_aggregate.py
python -u 103_aggregate_per_month.py
python -u 104_aggregate_per_month_summary.py
python -u 105_date.py
python -u 106_aggregate.py
python -u 107_date.py
python -u 108_aggregate.py
python -u 109_aggregate.py

# =============================================================================
# 200 new_merchant
# =============================================================================
python -u 201_init.py
python -u 202_aggregate.py
python -u 203_aggregate_per_month.py
python -u 204_aggregate_per_month_summary.py
python -u 205_date.py
python -u 206_aggregate.py
python -u 207_date.py
python -u 208_aggregate.py
python -u 209_aggregate.py

# =============================================================================
# 300 merchants
# =============================================================================
python -u 301_init.py
python -u 302_historical_aggregate.py
python -u 303_new_aggregate.py
python -u 304_union_aggregate.py
python -u 305_unique_aggregate.py
python -u 306_merchants_score.py

# =============================================================================
# 400 union
# =============================================================================
python -u 401_init.py

python -u 002_base.py # TODO: change

python -u 402_aggregate.py
python -u 403_aggregate_per_month.py
python -u 404_date.py
python -u 405_aggregate.py
python -u 406_date.py
python -u 407_aggregate_per_month.py
python -u 408_aggregate.py
python -u 409_aggregate.py
python -u 410_repurchase.py
python -u 411_authorized.py