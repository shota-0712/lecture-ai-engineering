import pytest
import time
import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def load_model():
    """モデルをロードする関数"""
    # day5/演習1 のモデルを使用
    model_path = os.path.join("day5", "演習1", "models", "titanic_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_test_data():
    """テストデータをロードする関数"""
    # day5/演習1 のデータを使用
    data_path = os.path.join("day5", "演習1", "data", "test.csv")
    
    # データを読み込む
    df = pd.read_csv(data_path)
    
    # 前処理
    if 'Survived' in df.columns:
        X = df.drop(['Survived'], axis=1)
        y = df['Survived']
    else:
        X = df
        # ダミーのラベル
        y = pd.Series([0, 1] * (len(df) // 2) + [0] * (len(df) % 2))
    
    # カテゴリカル変数の処理
    X = pd.get_dummies(X)
    
    return X, y

def test_model_inference_accuracy():
    """モデルの推論精度をテスト"""
    model = load_model()
    X, y = load_test_data()