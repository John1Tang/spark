# Spark ML 特征工程：从原始数据到模型输入

> 本文档深度解读 Spark ML 特征工程的完整链路——如何从原始 DataFrame 列转换为 LogisticRegression 所需的 `Vector` 特征，以及每个 Transformer 的内部实现。

## 1. 特征工程全景

线性模型（如 LogisticRegression）的输入是一个单一的 `Vector` 列（默认名为 `features`）。而原始数据通常是多列的混合类型：

```
原始数据:
┌─────────┬──────────┬──────┬───────────────────┬────────┐
│ category│ text     │ age  │ purchase_history  │ label  │
│ (String)│ (String) │(Double)│ (Array<Double>) │(Double)│
├─────────┼──────────┼──────┼───────────────────┼────────┤
│ "A"     │ "hello"  │ 25.0 │ [1.0, 0.5, ...]   │ 1.0    │
│ "B"     │ "world"  │ 30.0 │ [0.0, 1.2, ...]   │ 0.0    │
└─────────┴──────────┴──────┴───────────────────┴────────┘

↓ 特征工程 Pipeline

模型输入:
┌─────────────────────────────────────────────┬───────┐
│ features (Vector)                           │ label │
│ [1.0, 0.0, 0.5, 0.3, ..., 25.0, 1.0, ...]  │ 1.0   │
└─────────────────────────────────────────────┴───────┘
```

## 2. 特征 Transformers 分类

### 2.1 类别特征编码

| Transformer | 输入 | 输出 | 说明 |
|------------|------|------|------|
| `StringIndexer` | String | Double (index) | 字符串 → 数值索引 |
| `OneHotEncoder` | Double (index) | Vector (sparse) | 索引 → 独热向量 |
| `TargetEncoder` | String + label | Double | 目标编码（均值编码） |

### 2.2 文本特征

| Transformer | 输入 | 输出 | 说明 |
|------------|------|------|------|
| `Tokenizer` | String | Array[String] | 分词 |
| `HashingTF` | Array[String] | Vector (sparse) | Hash trick |
| `IDF` | Vector (TF) | Vector (weighted) | 逆文档频率 |
| `CountVectorizer` | Array[String] | Vector (count) | 词频统计 |
| `Word2Vec` | Array[String] | Vector (embedding) | 词向量 |

### 2.3 数值特征处理

| Transformer | 输入 | 输出 | 说明 |
|------------|------|------|------|
| `StandardScaler` | Vector | Vector | 标准化 (z-score) |
| `MinMaxScaler` | Vector | Vector | 缩放到 [min, max] |
| `Normalizer` | Vector | Vector | L1/L2/L∞ 归一化 |
| `RobustScaler` | Vector | Vector | 中位数 + IQR 缩放 |
| `MaxAbsScaler` | Vector | Vector | 最大绝对值缩放 |

### 2.4 特征组合与选择

| Transformer | 输入 | 输出 | 说明 |
|------------|------|------|------|
| `VectorAssembler` | 多列 | Vector | 拼接多个列 |
| `VectorIndexer` | Vector | Vector | 自动识别类别特征 |
| `Interaction` | 多列 | Vector | 特征交叉 |
| `PolynomialExpansion` | Vector | Vector | 多项式展开 |
| `ChiSqSelector` | Vector + label | Vector | 卡方选择 |
| `FeatureHasher` | Map/多列 | Vector | 特征 Hash |

### 2.5 降维

| Transformer | 输入 | 输出 | 说明 |
|------------|------|------|------|
| `PCA` | Vector | Vector | 主成分分析 |
| `BucketedRandomProjectionLSH` | Vector | Vector | 随机投影 |

## 3. VectorAssembler：特征拼接核心

[VectorAssembler.scala:44](mllib/src/main/scala/org/apache/spark/ml/feature/VectorAssembler.scala#L44):

`VectorAssembler` 是特征工程的**最后一公里**——将多个列拼接成一个 `Vector`。

### 3.1 工作原理

```scala
// 用户配置
new VectorAssembler()
  .setInputCols(Array("categoryVec", "tfidfVec", "age", "purchaseVec"))
  .setOutputCol("features")

// 内部处理
override def transform(dataset: Dataset[_]): DataFrame = {
  // 1. 获取每个输入列的 schema 信息
  val inputColsWithField = $(inputCols).map { c =>
    c -> SchemaUtils.getSchemaField(schema, c)
  }

  // 2. 获取 Vector 列的长度（从 Metadata 中）
  val vectorColsLengths = VectorAssembler.getLengths(dataset, vectorCols, ...)

  // 3. 构建输出 schema 的 AttributeGroup（列名映射）
  val featureAttributes = ...  // 保留每个特征的来源信息

  // 4. 生成 UDF 拼接 Vector
  val assembleUDF = udf { row: Row =>
    // 将每个输入值转为 Double 或 Vector 元素，拼接成一个 DenseVector
    Vectors.dense(assembledValues)
  }

  dataset.withColumn($(outputCol), assembleUDF(struct(inputCols: _*)))
}
```

### 3.2 支持的输入类型

```scala
inputCol.dataType match {
  case DoubleType | FloatType | IntegerType | LongType | ShortType | ByteType =>
    // 标量 → 单个 Double 值
  case BooleanType =>
    // true → 1.0, false → 0.0
  case VectorUDT =>
    // 展开为多个 Double 值
}
```

### 3.3 关键设计：AttributeGroup

拼接后的 Vector 保留了**元数据**（Metadata），记录了每个特征来自哪个原始列：

```scala
val attributeGroup = new AttributeGroup("features", featureAttributes.toArray)
```

这在模型解释和特征重要性分析中非常有用。例如，训练完 LR 后可以追溯每个系数对应的原始特征名。

## 4. StringIndexer + OneHotEncoder：类别编码

这是最常见的类别特征处理链路：

```scala
// Step 1: String → Index
val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")
  .setStringOrderType("frequencyAsc")  // 按频率排序，最少见的=0

// Step 2: Index → OneHot
val encoder = new OneHotEncoder()
  .setInputCol("categoryIndex")
  .setOutputCol("categoryVec")
  .setDropLast(true)  // 去掉最后一列（避免共线性）
```

### 4.1 StringIndexer 实现

```scala
override def fit(dataset: Dataset[_]): StringIndexerModel = {
  // 1. 统计每个值的频率
  val labels = dataset.select(col($(inputCol)))
    .groupBy(col($(inputCol)).cast(StringType).as("label"))
    .count()
    .collect()

  // 2. 排序（按频率、字母序等策略）
  val labelsSorted = $(stringOrderType) match {
    case "frequencyDesc" => labels.sortBy(-_.getAs[Long](1))
    case "frequencyAsc"  => labels.sortBy(_.getAs[Long](1))
    case "alphabetDesc"  => labels.sortBy(-_.getString(0))
    case "alphabetAsc"   => labels.sortBy(_.getString(0))
  }

  // 3. 构建 label → index 映射
  val labelsArray = labelsSorted.map(_.getString(0))
  new StringIndexerModel(uid, labelsArray, ...)
}

override def transform(dataset: Dataset[_]): DataFrame = {
  val indexer = udf { (label: String) =>
    labelToIndex.getOrElse(label, handleInvalid(label))
  }
  dataset.withColumn($(outputCol), indexer(col($(inputCol))))
}
```

### 4.2 OneHotEncoder 实现

```scala
// StringIndexerModel 产出的 index 是 0, 1, 2, ..., K-1
// OneHotEncoder 将其转为长度为 K (或 K-1) 的 SparseVector

override def transform(dataset: Dataset[_]): DataFrame = {
  val encodeUDF = udf { (index: Double) =>
    if (index >= 0 && index < numCategories) {
      val idx = index.toInt
      if ($(dropLast) && idx == numCategories - 1) {
        // dropLast=true 且是最后一个类别 → 全零向量
        Vectors.sparse(numCategories - 1, Seq.empty)
      } else {
        val size = if ($(dropLast)) numCategories - 1 else numCategories
        Vectors.sparse(size, Seq(idx -> 1.0))
      }
    } else {
      handleInvalid(index)
    }
  }
  dataset.withColumn($(outputCol), encodeUDF(col($(inputCol))))
}
```

**为什么 `dropLast=true`？**

在线性模型中，如果使用完整的独热编码（K 个类别产生 K 列），会产生**完全共线性**问题（所有独热列之和等于截距列）。去掉一列后，剩余 K-1 列是线性无关的。

**注意**：Spark 4.x 的 `OneHotEncoder` 支持多列输入：

```scala
new OneHotEncoder()
  .setInputCols(Array("categoryIndex1", "categoryIndex2"))
  .setOutputCols(Array("categoryVec1", "categoryVec2"))
```

## 5. HashingTF + IDF：文本特征

这是 NLP 场景的经典特征提取链路：

```scala
// Step 1: 分词
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

// Step 2: Hash trick (Term Frequency)
val hashingTF = new HashingTF()
  .setInputCol("words")
  .setOutputCol("rawFeatures")
  .setNumFeatures(10000)  // 特征维度

// Step 3: 逆文档频率
val idf = new IDF()
  .setInputCol("rawFeatures")
  .setOutputCol("features")
  .setMinDocFreq(2)  // 至少出现在 2 个文档中的词才保留
```

### 5.1 HashingTF 实现

```scala
override def transform(dataset: Dataset[_]): DataFrame = {
  val transformUDF = udf { (words: Seq[_]) =>
    val termFrequencies = mutable.Map.empty[Int, Double]
    words.foreach { term =>
      val idx = nonNegativeMod(hashFunc(term), $(numFeatures))
      termFrequencies(idx) = termFrequencies.getOrElse(idx, 0.0) + 1.0
    }
    Vectors.sparse($(numFeatures), termFrequencies.toSeq)
  }
  dataset.withColumn($(outputCol), transformUDF(col($(inputCol))))
}
```

**Hash Trick 的优劣**：
- **优点**：不需要预先构建词表，内存可控，支持流式处理
- **缺点**：Hash 碰撞（不同词映射到同一位置），无法追溯原始词

### 5.2 IDF 实现

```scala
// fit: 计算每个词的 IDF 权重
override def fit(dataset: Dataset[_]): IDFModel = {
  val n = dataset.count().toDouble
  val df = dataset.select(col($(inputCol)))
    .flatMap { case Row(vec: Vector) =>
      vec.toArray.zipWithIndex.collect { case (v, i) if v > 0.0 => i }
    }
    .distinct()
    .groupBy($"value".as("index"))
    .count()

  // IDF(i) = log((n + 1) / (df(i) + 1)) + 1  (smoothed)
  val idf = df.map { case Row(index: Int, count: Long) =>
    index -> (math.log((n + 1.0) / (count + 1.0)) + 1.0)
  }.collect().toMap

  new IDFModel(uid, Vectors.sparse($(numFeatures), idf.toSeq))
}

// transform: TF * IDF
override def transform(dataset: Dataset[_]): DataFrame = {
  val transformUDF = udf { (tf: Vector) =>
    val values = tf.toArray.zipWithIndex.map { case (v, i) =>
      v * idf(i)  // TF-IDF
    }
    Vectors.dense(values)
  }
  dataset.withColumn($(outputCol), transformUDF(col($(inputCol))))
}
```

## 6. StandardScaler：特征标准化

[StandardScaler.scala](mllib/src/main/scala/org/apache/spark/ml/feature/StandardScaler.scala):

### 6.1 实现

```scala
// fit: 计算均值和标准差
override def fit(dataset: Dataset[_]): StandardScalerModel = {
  val summary = dataset.select(col($(inputCol)))
    .rdd
    .treeAggregate(new MultivariateOnlineSummarizer())(
      (agg, v) => agg.add(v.asInstanceOf[Vector]),
      (agg1, agg2) => agg1.merge(agg2)
    )
  new StandardScalerModel(uid, summary.mean, summary.variance)
}

// transform: z-score 标准化
override def transform(dataset: Dataset[_]): DataFrame = {
  val transformUDF = udf { (vec: Vector) =>
    val values = vec.toArray
    var i = 0
    while (i < values.length) {
      if ($(withStd) && std(i) != 0.0) {
        values(i) = (values(i) - mean(i)) / std(i)  // z-score
      } else if ($(withMean)) {
        values(i) = values(i) - mean(i)              // 只中心化
      }
      i += 1
    }
    Vectors.dense(values)
  }
  dataset.withColumn($(outputCol), transformUDF(col($(inputCol))))
}
```

### 6.2 与 LR 训练内部标准化的关系

注意：`StandardScaler`（特征工程 Transformer）和 LR 训练**内部的标准化**是两个不同的东西：

| | StandardScaler (Transformer) | LR 内部标准化 |
|---|---|---|
| 目的 | 持久化标准化后的特征 | 加速优化收敛 |
| 输出 | 持久化到 DataFrame | 仅训练时内存中使用 |
| 推理 | 需要显式应用 | 不需要（系数已反标准化） |
| 用户可见 | 是 | 否 |

**最佳实践**：如果在 Pipeline 中使用了 `StandardScaler`，应该设置 LR 的 `standardization=false`，避免双重标准化：

```scala
val pipeline = new Pipeline().setStages(Array(
  new StandardScaler().setInputCol("rawFeatures").setOutputCol("features"),
  new LogisticRegression().setStandardization(false)  // ← 重要！
))
```

## 7. RFormula：一站式特征工程

`RFormula` 是 Spark ML 对 R 语言公式语法的实现，可以自动完成所有特征工程：

```scala
val formula = new RFormula()
  .setFormula("label ~ country + hour + clicked")
  //              ↑ 目标   ↑ 类别    ↑ 数值  ↑ 已有特征
  .setFeaturesCol("features")
  .setLabelCol("label")

// 等价于手动执行：
// 1. StringIndexer on "country"
// 2. OneHotEncoder on "countryIndex"
// 3. VectorAssembler of ["countryVec", "hour", "clicked"]
```

**内部实现**：`RFormula` 解析公式语法树，自动生成对应的 Transformer 链。它支持：
- 数值列：直接使用
- 字符串列：自动 StringIndexer + OneHotEncoder
- 交互项：`country:hour` → 交叉特征
- 公式操作符：`+`, `-`, `:`, `*`, `.`

## 8. 特征选择

### 8.1 ChiSqSelector

基于卡方统计量的特征选择：

```scala
val selector = new ChiSqSelector()
  .setNumTopFeatures(50)        // 选择前 50 个最有区分力的特征
  .setFeaturesCol("features")
  .setLabelCol("label")
  .setOutputCol("selectedFeatures")
```

**实现原理**：
1. 对每个特征计算卡方统计量 `χ² = Σ (observed - expected)² / expected`
2. 按 χ² 值排序，选择 top-K

```scala
override def fit(dataset: Dataset[_]): ChiSqSelectorModel = {
  // 计算每个特征的卡方统计量和 p-value
  val chiSqStats = ChiSquareTest.test(dataset, $(featuresCol), $(labelCol))
  // 选择 top-K
  val selectedFeatures = topKIndices(chiSqStats, $(numTopFeatures))
  new ChiSqSelectorModel(uid, selectedFeatures)
}
```

### 8.2 VarianceThresholdSelector

基于方差的特征选择（去除常数或近常数特征）：

```scala
val selector = new VarianceThresholdSelector()
  .setVarianceThreshold(0.01)  // 去除方差 < 0.01 的特征
```

## 9. 特征工程 Pipeline 完整示例

```scala
// 原始数据
// ┌──────┬──────────────┬─────┬───────┬───────┐
// │gender│       comment│  age│  score│ label │
// ├──────┼──────────────┼─────┼───────┼───────┤
// │   "M"│"great product"│  25 │   4.5 │   1.0 │
// │   "F"│"not bad"     │  30 │   3.2 │   0.0 │
// └──────┴──────────────┴─────┴───────┴───────┘

val pipeline = new Pipeline().setStages(Array(
  // 1. 类别编码
  new StringIndexer()
    .setInputCol("gender")
    .setOutputCol("genderIndex"),

  new OneHotEncoder()
    .setInputCol("genderIndex")
    .setOutputCol("genderVec"),

  // 2. 文本特征
  new Tokenizer()
    .setInputCol("comment")
    .setOutputCol("words"),

  new HashingTF()
    .setInputCol("words")
    .setOutputCol("rawTF")
    .setNumFeatures(5000),

  // 3. IDF 加权（需要 fit）
  new IDF()
    .setInputCol("rawTF")
    .setOutputCol("tfidfVec"),

  // 4. 特征拼接
  new VectorAssembler()
    .setInputCols(Array("genderVec", "tfidfVec", "age", "score"))
    .setOutputCol("rawFeatures"),

  // 5. 标准化
  new StandardScaler()
    .setInputCol("rawFeatures")
    .setOutputCol("features")
    .setWithStd(true)
    .setWithMean(true),

  // 6. 逻辑回归
  new LogisticRegression()
    .setStandardization(false)  // 已经标准化过了
    .setRegParam(0.01)
    .setMaxIter(100)
))

// 训练
val model = pipeline.fit(trainingData)

// 推理
val predictions = model.transform(testData)
```

这个 Pipeline 的 fitted model 结构：
```
PipelineModel
├── StringIndexerModel (gender → genderIndex)
├── OneHotEncoderModel (genderIndex → genderVec)
├── Tokenizer (comment → words)
├── HashingTFModel (words → rawTF)
├── IDFModel (rawTF → tfidfVec)
├── VectorAssemblerModel (多列 → rawFeatures)
├── StandardScalerModel (rawFeatures → features)
└── LogisticRegressionModel (features → prediction)
```

## 10. 特征工程的性能注意事项

### 10.1 VectorAssembler 的列顺序

`VectorAssembler` 按 `inputCols` 的顺序拼接特征。这个顺序决定了最终特征向量中每个维度的含义。建议在命名中体现顺序：

```scala
.setInputCols(Array(
  // 前 3 维: gender (one-hot, 3 categories → 2 dims after dropLast)
  "genderVec",
  // 第 4-5003 维: comment (tfidf, 5000 dims)
  "tfidfVec",
  // 第 5004 维: age
  "age",
  // 第 5005 维: score
  "score"
))
```

训练完后，可以通过模型系数的索引回溯到原始特征：
```scala
val coefArray = model.coefficientMatrix.toArray
// coefArray[0..1] → genderVec
// coefArray[2..5001] → tfidfVec
// coefArray[5002] → age
// coefArray[5003] → score
```

### 10.2 稀疏 vs 稠密向量

| | 稀疏向量 (SparseVector) | 稠密向量 (DenseVector) |
|---|---|---|
| 存储 | 仅存非零值 + 索引 | 全部值 |
| 适用 | 文本特征、独热编码 | 数值特征 |
| BLAS 加速 | 部分操作支持 | 全部支持 |
| 内存 | O(nnz) | O(n) |

`VectorAssembler` 的输出始终是 `DenseVector`。如果大部分特征是稀疏的，考虑使用 `FeatureHasher` 代替：

```scala
new FeatureHasher()
  .setInputCols(Array("gender", "comment", "category"))
  .setOutputCol("features")
  .setNumFeatures(10000)  // 输出 SparseVector
```
