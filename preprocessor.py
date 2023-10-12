import pandas as pd
import numpy as np


def make_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    drop_col = ["騎手", "日付", "開催", "レース名", "枠番", "馬番", "着順", "着差", "通過", "ペース", "上り", "勝ち馬(2着馬)", "賞金"]
    df = df.drop(drop_col, axis=1)

    df.replace(" ", np.nan, inplace=True)
    df.replace('\xa0', np.nan, inplace=True)
    df.replace("計不", np.nan, inplace=True)

    df = df.dropna()

    df["タイム"] = df["タイム"].apply(lambda x : int(x.split(":")[0]) * 60 + float(x.split(":")[1]))
    # '距離'列を2つに分割
    df[['地面データ', '距離データ']] = df['距離'].str.extract(r'(\D+)(\d+)')
    # 不要な列を削除
    df = df.drop(columns=['距離'])
    # 正規表現を使用して"体重"列を分割
    df[['馬体重データ', '重量増減データ']] = df['馬体重'].str.extract(r'(\d+)\(([+-]?\d+)\)')
    # 不要な列を削除
    df = df.drop(columns=['馬体重'])

    df = pd.get_dummies(df, columns=["天気", "馬場", "地面データ"])

    return df
