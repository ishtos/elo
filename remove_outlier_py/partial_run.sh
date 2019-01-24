# =============================================================================
# 000 initializer
# =============================================================================
# python -u 001_preprocess.py
# python -u 002_init.py

# =============================================================================
# 100 historical
# =============================================================================
# python -u 101_init.py

# python -u 102_aggregate.py
# python -u 103_aggregate_per_month.py

# python -u 111_authorized_aggregate.py
# python -u 112_authorized_aggregate_per_month.py

# python -u 121_authorized_rate.py
# python -u 122_repurchase_rate.py

# python -u 131_pivot_table_purchase_amount.py
python -u 132_pivot_table_installments.py

# =============================================================================
# 200 new_merchant
# =============================================================================
# python -u 201_init.py

# python -u 202_aggregate.py
# python -u 203_aggregate_per_month.py

# python -u 211_authorized_aggregate.py
# python -u 212_authorized_aggregate_per_month.py

# python -u 231_pivot_table_purchase_amount.py
python -u 232_pivot_table_installments.py

# =============================================================================
# 300 merchants
# =============================================================================
# python -u 301_init.py
# python -u 302_historical_aggregate.py
# python -u 303_new_aggregate.py
# python -u 304_union_aggregate.py
# python -u 305_unique_aggregate.py
# python -u 306_merchants_score.py

# =============================================================================
# 400 union
# =============================================================================
# python -u 401_init.py

# python -u 402_aggregate.py
# python -u 403_aggregate_per_month.py

# python -u 411_authorized_aggregate.py
# python -u 412_authorized_aggregate_per_month.py

# python -u 421_authorized_rate.py
# python -u 422_repurchase_rate.py

# =============================================================================
# 600 CV
# =============================================================================
# python -u 601_preprocess.py

# =============================================================================
# 700 CV
# =============================================================================
# python -u 701_lgb_cv.py