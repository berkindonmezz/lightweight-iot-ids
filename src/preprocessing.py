import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_COLS,
    DROP_COLS,
    META_COL,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TARGET_COL,
)


def load_raw_dataset() -> pd.DataFrame:
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{RAW_DATA_DIR} klasöründe CSV dosyası bulunamadı.")
    return pd.read_csv(csv_files[0], low_memory=False)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Ham Edge-IIoTset verisini sızıntısız ve tutarlı hale getirir."""
    stats = {"raw_rows": int(len(df)), "raw_cols": int(df.shape[1])}

    # 1) Sistematik sütun kayması tespit edilen MITM sınıfını çıkar
    mitm_mask = df[META_COL] == "MITM"
    stats["removed_corrupted_mitm_rows"] = int(mitm_mask.sum())
    df = df.loc[~mitm_mask].copy()

    # 2) Sızıntı, gürültü ve sıfır varyans sütunlarını çıkar
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    stats["dropped_cols_count"] = len(cols_to_drop)

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, META_COL]]
    stats["remaining_feature_count"] = len(feature_cols)

    # 3) Metin sütunlarında '0.0' / '0' ikiliğini gider
    str_cols = [
        "http.request.method",
        "http.request.version",
        "dns.qry.name.len",
        "mqtt.conack.flags",
        "mqtt.protoname",
    ]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().replace({"0.0": "0"})

    # http.request.version içindeki taşmış 5 satırı standartlaştır
    df["http.request.version"] = df["http.request.version"].where(
        df["http.request.version"].isin(["0", "HTTP/1.1", "HTTP/1.0"]),
        "OTHER",
    )

    # dns.qry.name.len içindeki alan adlarını gerçek karakter uzunluğuna çevir
    df["dns.qry.name.len"] = df["dns.qry.name.len"].apply(
        lambda x: 0 if x == "0" else len(x)
    ).astype(np.int32)

    # Sadece 2 değer alan MQTT metin alanlarını temiz ikili (0/1) sayıya çevir
    df["mqtt.conack.flags"] = (df["mqtt.conack.flags"] != "0").astype(np.int8)
    df["mqtt.protoname"] = (df["mqtt.protoname"] != "0").astype(np.int8)

    # tcp.srcport'u sayısal tipe dönüştür
    df["tcp.srcport"] = pd.to_numeric(df["tcp.srcport"], errors="raise")

    # Hedef sütunu tamsayı yap
    df[TARGET_COL] = df[TARGET_COL].astype(np.int8)

    # 4) Özellik + Attack_label bazında birebir kopya satırları temizle
    before_dedup = len(df)
    df = df.drop_duplicates(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    stats["removed_exact_duplicates"] = int(before_dedup - len(df))

    # 5) Tüm özellikleri aynı ama etiketi çelişkili (hem 0 hem 1) satırları temizle
    conflict_mask = df.duplicated(subset=feature_cols, keep=False)
    stats["removed_conflicting_rows"] = int(conflict_mask.sum())
    df = df.loc[~conflict_mask].reset_index(drop=True)

    stats["final_clean_rows"] = int(len(df))
    stats["feature_cols"] = feature_cols
    return df, feature_cols, stats


def split_and_save(df: pd.DataFrame, stats: dict) -> None:
    """Attack_type üzerinden Stratified %70 Train / %15 Val / %15 Test ayrımı yapar."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # %70 Train, %30 Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[META_COL],
        random_state=RANDOM_STATE,
    )

    # %30 Temp'i yarı yarıya böl -> %15 Val, %15 Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[META_COL],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_parquet(PROCESSED_DATA_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DATA_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DATA_DIR / "test.parquet", index=False)

    stats["splits"] = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_attack_ratio": round(float(train_df[TARGET_COL].mean()), 4),
        "val_attack_ratio": round(float(val_df[TARGET_COL].mean()), 4),
        "test_attack_ratio": round(float(test_df[TARGET_COL].mean()), 4),
    }

    with open(RESULTS_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("=== VERİ TEMİZLEME VE BÖLME TAMAMLANDI ===")
    print(f"Ham satır sayısı        : {stats['raw_rows']:,}")
    print(f"Çıkarılan bozuk MITM    : {stats['removed_corrupted_mitm_rows']:,}")
    print(f"Çıkarılan tam kopya     : {stats['removed_exact_duplicates']:,}")
    print(f"Çıkarılan çelişkili     : {stats['removed_conflicting_rows']:,}")
    print(f"Temiz son satır sayısı  : {stats['final_clean_rows']:,}")
    print(f"Özellik sayısı          : {stats['remaining_feature_count']}")
    print(f"Train / Val / Test      : {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
    print(
        f"Saldırı oranı (T/V/Ts)  : "
        f"%{stats['splits']['train_attack_ratio']*100:.2f} / "
        f"%{stats['splits']['val_attack_ratio']*100:.2f} / "
        f"%{stats['splits']['test_attack_ratio']*100:.2f}"
    )
    print(f"\nDosyalar kaydedildi -> {PROCESSED_DATA_DIR}")


def build_preprocessor(feature_cols: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    """
    Faz 3 ve Faz 4'te sadece X_train üzerinde fit edilecek sızıntısız Pipeline bileşeni.
    Ağaç tabanlı modeller için scale_numeric=False, doğrusal modeller için True verilir.
    """
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="0")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    raw_df = load_raw_dataset()
    clean_df, feat_cols, summary_stats = clean_dataset(raw_df)
    split_and_save(clean_df, summary_stats)