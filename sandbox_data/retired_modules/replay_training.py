from __future__ import annotations

"""Nightly Spark job for failure replay training."""

from dataclasses import dataclass
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier


@dataclass
class TrainingConfig:
    log_path: str
    model_path: str


class ReplayTrainer:
    """Aggregate error->fix->success sequences and train a GBT model."""

    def __init__(self, spark: SparkSession | None = None) -> None:
        if spark is None:
            self.spark = (
                SparkSession.builder
                .appName("replay-training")
                .master("local[*]")
                .config("spark.driver.host", "127.0.0.1")
                .getOrCreate()
            )
        else:
            self.spark = spark

    def load_logs(self, path: str):
        return self.spark.read.json(path)

    def build_sequences(self, df):
        w = Window.partitionBy("fingerprint").orderBy("ts")
        df = df.withColumn("prev", lag(col("state")).over(w))
        return df.filter((col("prev") == "fix") & (col("state") == "success"))

    def train(self, config: TrainingConfig) -> Any:
        df = self.load_logs(config.log_path)
        seq = self.build_sequences(df)
        feats = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
        data = feats.transform(seq).select(col("label"), col("features"))
        train, _ = data.randomSplit([0.8, 0.2], seed=42)
        clf = GBTClassifier(labelCol="label", featuresCol="features", maxIter=20)
        model = clf.fit(train)
        model.write().overwrite().save(config.model_path)
        return model


__all__ = ["TrainingConfig", "ReplayTrainer"]
