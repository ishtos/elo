historical_transactions.csv
new_merchant_transactions.csv.gz
カラムが全て同じなので、同様に扱って処理すル方が良いのか？
特に同じに扱って、時間毎に分析した方が良さそう...。

merchants.csv
historicalなどとの組み合わせ方
店毎の人気や売り上げを作り、店の信頼度を計算する？

category_2 == region

historical_transactionsのcategory_3は、installmentsの区別
0:0=?
1:0=一欠損血
2:2~12,999=分割払い(999は12回以上の分割払い?)
3:-1=欠損値