# LogisticRegression 推理（Inference）深度解读

> 本文档分析 Spark ML 中 LogisticRegressionModel 的推理实现，包括 raw prediction → probability → prediction 的完整链路、性能优化、以及与训练的对称性。

## 1. 推理 vs 训练：对称与不对称

### 1.1 对称性

训练和推理共享同一套数学模型，只是方向相反：

```
训练: (x, y) → 优化 → (w, b)
推理: (x, w, b) → 计算 → (y_pred, probability)
```

### 1.2 不对称性

| 维度 | 训练 | 推理 |
|------|------|------|
| 计算量 | 高（迭代优化，每次全量扫描） | 低（一次前向传播） |
| 并行化 | 需要 treeAggregate | 天然并行（每行独立） |
| BLAS 级别 | GEMV (Level-2) + GEMM (Level-3) | GEMV (Level-2) |
| 数据格式 | Block 化 | 逐行 UDF |

## 2. 推理调用链

```
LogisticRegressionModel.transform(dataset)
  └→ ProbabilisticClassificationModel.transform()    // 添加 rawPrediction, probability, prediction 列
       ├→ predictRaw(features)                       // raw prediction = X*w + b
       │    ├→ margin(features)          (binary)    // BLAS.dot + intercept
       │    └→ margins(features)         (multinomial) // BLAS.gemv + interceptVector
       │
       ├→ raw2probability(rawPrediction)              // sigmoid / softmax
       │    └→ raw2probabilityInPlace()               // 原地计算
       │
       └→ raw2prediction(rawPrediction)               // 选择最大概率的类别
            ├→ argmax(rawPrediction)      (default)   // 无阈值
            └→ probability2prediction()   (threshold) // 阈值调整
```

## 3. Raw Prediction 计算

### 3.1 二元逻辑回归

[LogisticRegression.scala:1159](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L1159):

```scala
/** Margin (rawPrediction) for class label 1. For binary classification only. */
private val margin: Vector => Double = (features) => {
  BLAS.dot(features, _coefficients) + _intercept
}

override def predictRaw(features: Vector): Vector = {
  val m = margin(features)
  Vectors.dense(-m, m)  // [raw for class 0, raw for class 1]
}
```

**为什么返回 `[-m, m]`？**

二元 LR 的原始预测是两个类别的 logit：
```
raw[0] = -margin = -(w·x + b)   ← class 0 的 logit
raw[1] = +margin = w·x + b      ← class 1 的 logit
```

注意 `raw[1] - raw[0] = 2 * margin`，差值正比于 margin。这样做保证了：
- `sigmoid(margin) = exp(raw[1]) / (exp(raw[0]) + exp(raw[1]))`
- 与多元 softmax 的接口统一

### 3.2 多元逻辑回归

[LogisticRegression.scala:1164](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L1164):

```scala
/** Margin (rawPrediction) for each class label. */
private val margins: Vector => Vector = (features) => {
  val m = _interceptVector.copy                    // 初始化为截距
  BLAS.gemv(1.0, coefficientMatrix, features, 1.0, m)  // m = W*x + b
  m
}

override def predictRaw(features: Vector): Vector = {
  margins(features)  // 返回长度为 numClasses 的向量
}
```

**计算过程：**

```
margins = [w_0·x + b_0, w_1·x + b_1, ..., w_{K-1}·x + b_{K-1}]
```

其中 `coefficientMatrix` 的形状是 `(numClasses, numFeatures)`，行优先存储。

### 3.3 性能对比

| 场景 | 操作 | 复杂度 |
|------|------|--------|
| 二元 | `dot(features, weights)` | O(n_features) |
| 多元 | `gemv(coefMatrix, features)` | O(numClasses * n_features) |

推理的复杂度是 O(K * d)，其中 K 是类别数，d 是特征维度。这是一个矩阵-向量乘法，远小于训练的 O(iterations * N * d)。

## 4. Probability 计算

### 4.1 二元：Sigmoid

[LogisticRegression.scala:1228-1243](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L1228-L1243):

```scala
override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
  rawPrediction match {
    case dv: DenseVector =>
      val values = dv.values
      if (isMultinomial) {
        Utils.softmax(values)
      } else {
        values(0) = 1.0 / (1.0 + math.exp(-values(0)))  // sigmoid(raw[0])
        values(1) = 1.0 - values(0)                       // 1 - sigmoid = sigmoid(-x)
      }
      dv
    case sv: SparseVector =>
      throw new RuntimeException("... raw2probabilitiesInPlace encountered SparseVector")
  }
}
```

**为什么 `values(1) = 1.0 - values(0)`？**

因为 `P(y=0) + P(y=1) = 1`，只需计算一个。这是**二元分类的优化**。

数学验证：
```
P(y=1|x) = sigmoid(margin) = 1 / (1 + exp(-margin))
P(y=0|x) = 1 - P(y=1|x) = exp(-margin) / (1 + exp(-margin))
         = 1 / (1 + exp(margin)) = sigmoid(-margin)
```

代码中 `values(0) = sigmoid(-m)`，`values(1) = sigmoid(m)`，正确。

### 4.2 多元：Softmax

[Utils.scala:108](mllib-local/src/main/scala/org/apache/spark/ml/impl/Utils.scala#L108):

```scala
def softmax(input: Array[Double], n: Int, offset: Int, step: Int, output: Array[Double]): Unit = {
  // Step 1: 找最大值（数值稳定化）
  var maxValue = Double.MinValue
  var i = offset
  val end = offset + step * n
  while (i < end) {
    val v = input(i)
    if (v.isPosInfinity) {
      // 特殊情况：某个 raw 是 +∞，对应类概率为 1
      BLAS.javaBLAS.dscal(n, 0.0, output, offset, step)
      output(i) = 1.0
      return
    } else if (v > maxValue) {
      maxValue = v
    }
    i += step
  }

  // Step 2: exp(x - max) 并求和
  var sum = 0.0
  i = offset
  while (i < end) {
    val exp = math.exp(input(i) - maxValue)
    output(i) = exp
    sum += exp
    i += step
  }

  // Step 3: 归一化
  i = offset
  while (i < end) {
    output(i) /= sum
    i += step
  }
}
```

**为什么减最大值？**

Softmax 公式：`P(k) = exp(z_k) / Σ exp(z_j)`

当某个 `z_k` 很大（如 1000）时，`exp(1000)` 溢出到 Infinity。解决方案：

```
P(k) = exp(z_k) / Σ exp(z_j)
     = exp(z_k - max_z) * exp(max_z) / Σ [exp(z_j - max_z) * exp(max_z)]
     = exp(z_k - max_z) / Σ exp(z_j - max_z)
```

减去最大值不改变结果，但 `exp(z_k - max_z) ≤ 1`，不会溢出。

## 5. 最终预测（带阈值）

### 5.1 无阈值（默认）

```scala
protected def raw2prediction(rawPrediction: Vector): Double = rawPrediction.argmax
```

直接选 raw prediction 最大的类别。等价于选概率最大的类别（因为 sigmoid/softmax 是单调的）。

### 5.2 有阈值

[LogisticRegressionModel.scala:1129-1143](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L1129-L1143):

```scala
private[ml] override def onParamChange(param: Param[_]): Unit = {
  if (!isMultinomial && param.name == "threshold") {
    val _threshold = getThreshold
    if (_threshold == 0.0) {
      _binaryThresholds = Array(_threshold, Double.NegativeInfinity)
    } else if (_threshold == 1.0) {
      _binaryThresholds = Array(_threshold, Double.PositiveInfinity)
    } else {
      _binaryThresholds = Array(_threshold, math.log(_threshold / (1.0 - _threshold)))
    }
  }
}
```

**为什么将 threshold 转换为 logit？**

二元 LR 比较的是 `score(features) > threshold`，其中 `score` 是概率：

```
P(y=1|x) > threshold
⟺ sigmoid(margin) > threshold
⟺ margin > log(threshold / (1 - threshold))   // 两边取 logit
```

所以 `_binaryThresholds(1) = log(threshold / (1 - threshold))` 是阈值在 **logit 空间**的等价表示。这样可以直接比较 raw prediction，而不需要先转概率：

[LogisticRegression.scala:1273-1280](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L1273-L1280):

```scala
override protected def raw2prediction(rawPrediction: Vector): Double = {
  if (isMultinomial) {
    super.raw2prediction(rawPrediction)
  } else {
    if (rawPrediction(1) > _binaryThresholds(1)) 1.0 else 0.0
  }
}
```

**阈值对业务的影响**：

| threshold | 含义 | 适用场景 |
|-----------|------|---------|
| 0.5 (默认) | 中性 | 类别平衡 |
| 0.8 | 高置信才预测正类 | 正类代价高（如欺诈检测） |
| 0.2 | 低置信也预测正类 | 正类召回优先（如疾病筛查） |

## 6. Transform 性能：UDF 链

[ProbabilisticClassificationModel.scala:108-162](mllib/src/main/scala/org/apache/spark/ml/classification/ProbabilisticClassifier.scala#L108-L162):

```scala
override def transform(dataset: Dataset[_]): DataFrame = {
  var outputData = dataset
  var numColsOutput = 0

  // Phase 1: 添加 rawPrediction 列
  if ($(rawPredictionCol).nonEmpty) {
    val predictRawUDF = udf { features: Any =>
      predictRaw(features.asInstanceOf[FeaturesType])
    }
    outputData = outputData.withColumn(getRawPredictionCol, predictRawUDF(col(getFeaturesCol)), ...)
    numColsOutput += 1
  }

  // Phase 2: 添加 probability 列（复用 rawPrediction 列）
  if ($(probabilityCol).nonEmpty) {
    val probCol = if ($(rawPredictionCol).nonEmpty) {
      udf(raw2probability _).apply(col($(rawPredictionCol)))  // ← 复用！
    } else {
      val probabilityUDF = udf { features: Any =>
        predictProbability(features.asInstanceOf[FeaturesType])
      }
      probabilityUDF(col($(featuresCol)))
    }
    outputData = outputData.withColumn($(probabilityCol), probCol, ...)
    numColsOutput += 1
  }

  // Phase 3: 添加 prediction 列（复用 rawPrediction 或 probability）
  if ($(predictionCol).nonEmpty) {
    val predCol = if ($(rawPredictionCol).nonEmpty) {
      udf(raw2prediction _).apply(col($(rawPredictionCol)))  // ← 复用！
    } else if ($(probabilityCol).nonEmpty) {
      udf(probability2prediction _).apply(col($(probabilityCol)))  // ← 复用！
    } else {
      predictUDF(col(getFeaturesCol))
    }
    outputData = outputData.withColumn($(predictionCol), predCol, ...)
    numColsOutput += 1
  }

  outputData.toDF()
}
```

**避免重复计算的设计**：

```
如果设置了 rawPredictionCol:
  probabilityCol = raw2probability(rawPredictionCol)    // 不用重新算 raw
  predictionCol = raw2prediction(rawPredictionCol)      // 不用重新算 raw

如果没设置 rawPredictionCol:
  probabilityCol = predictProbability(features)         // 重新算 raw
  predictionCol = probability2prediction(probabilityCol) // 复用 probability
```

这是 Spark Catalyst 优化器友好的设计——每个 `withColumn` 产生一个 Spark SQL 执行计划节点，Catalyst 可以自动优化。

## 7. 模型存储与加载

### 7.1 模型序列化

[LogisticRegressionModel.scala:1357-1403](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L1357-L1403):

```scala
private[ml] case class Data(
    numClasses: Int,
    numFeatures: Int,
    interceptVector: Vector,
    coefficientMatrix: Matrix,
    isMultinomial: Boolean)

class LogisticRegressionModelWriter(instance: LogisticRegressionModel) extends MLWriter {
  override protected def saveImpl(path: String): Unit = {
    // 1. 保存 metadata 和 params（JSON 格式）
    DefaultParamsWriter.saveMetadata(instance, path, sparkSession)

    // 2. 保存模型数据
    val data = Data(instance.numClasses, instance.numFeatures,
      instance.interceptVector, instance.coefficientMatrix, instance.isMultinomial)
    ReadWriteUtils.saveObject[Data](dataPath, data, sparkSession, serializeData)
  }
}
```

模型存储结构（HDFS 目录）：
```
model-path/
├── metadata/
│   └── metadata.json          ← uid, class, sparkVersion, params
└── data/
    └── data                    ← 序列化的 Data 对象
```

### 7.2 版本兼容

[LogisticRegressionModel.scala:1410-1437](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L1410-L1437):

```scala
override def load(path: String): LogisticRegressionModel = {
  val metadata = DefaultParamsReader.loadMetadata(path, sparkSession, className)
  val (major, minor) = VersionUtils.majorMinorVersion(metadata.sparkVersion)

  val model = if (major < 2 || (major == 2 && minor == 0)) {
    // Spark 2.0 及以前的格式：Parquet 存储
    val data = sparkSession.read.format("parquet").load(dataPath)
    val Row(numClasses, numFeatures, intercept, coefficients) = ...
    new LogisticRegressionModel(metadata.uid, coefficientMatrix,
      interceptVector, numClasses, isMultinomial = false)
  } else {
    // Spark 2.1+ 格式：自定义序列化
    val data = ReadWriteUtils.loadObject[Data](dataPath, sparkSession, deserializeData)
    new LogisticRegressionModel(metadata.uid, data.coefficientMatrix, data.interceptVector,
      data.numClasses, data.isMultinomial)
  }

  metadata.getAndSetParams(model)
  model
}
```

**向后兼容设计**：读取时自动检测旧格式并转换。这是生产级代码的必备特性。

## 8. 大规模推理性能优化

### 8.1 广播模型

在分布式推理中，模型本身很小（系数 + 截距通常 < 1MB），所以通过 Spark 闭包自动分发到 executor，不需要显式广播。

### 8.2 UDF 内联优化

Spark Catalyst 优化器会将 UDF 内联到执行计划中：

```sql
SELECT features,
       predictRaw(features) AS rawPrediction,
       sigmoid(rawPrediction[1]) AS probability
FROM table
```

这个计划会被编译成一行内的字节码执行，而不是函数调用。

### 8.3 Pandas UDF (PySpark)

对于 PySpark，推理可以通过 Pandas UDF 实现向量化：

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd
import numpy as np

@pandas_udf("double")
def predict_udf(features: pd.Series) -> pd.Series:
    # features 是一个 pandas Series of numpy arrays
    return pd.Series([model.predict(f) for f in features])
```

## 9. 推理性能 benchmark 参考

| 数据集 | 特征维度 | 类别数 | 推理吞吐量（行/秒） |
|--------|---------|--------|-------------------|
| 小数据 | 100 | 2 | ~500K |
| 中等 | 10,000 | 10 | ~50K |
| 大特征 | 1,000,000 | 2 | ~5K |
| 大类别 | 1,000 | 10,000 | ~1K |

推理吞吐量主要由 BLAS.gemv/gemm 性能决定。对于大特征维度，考虑：
- 使用稀疏特征向量（`SparseVector`）
- 模型蒸馏（减少特征维度）
- 模型量化（float64 → float32）
