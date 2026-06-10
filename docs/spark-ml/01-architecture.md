# Spark ML 模块架构总览

> 本文档深度解读 Apache Spark ML（基于 DataFrame 的 ML Pipeline API）的整体架构设计，以 LogisticRegression 为主线贯穿分析。

## 1. 模块全景图

Spark ML 模块位于 `org.apache.spark.ml` 包下，代码在 `mllib/src/main/scala/` 目录中。注意：尽管包名叫 `mllib`，但 `org.apache.spark.ml` 是新一代基于 DataFrame 的高层 API（Spark 1.2+），而 `org.apache.spark.mllib` 是旧的基于 RDD 的底层 API。

```
org.apache.spark.ml/
├── Estimator.scala              # 训练器基类 (fit → Model)
├── Transformer.scala            # 转换器基类 (transform → DataFrame)
├── Pipeline.scala               # Pipeline 编排 (Estimator + Transformer 链)
├── Predictor.scala              # 预测问题抽象 (回归 + 分类的公共基类)
├── Model.scala                  # 模型基类
├── functions.scala              # DataFrame-friendly ML functions
├── events.scala                 # 事件系统
│
├── classification/              # 分类算法
│   ├── Classifier.scala
│   ├── ProbabilisticClassifier.scala
│   ├── LogisticRegression.scala      ★ 核心
│   ├── DecisionTreeClassifier.scala
│   ├── RandomForestClassifier.scala
│   ├── GBTClassifier.scala
│   ├── LinearSVC.scala
│   ├── NaiveBayes.scala
│   ├── MultilayerPerceptronClassifier.scala
│   ├── FMClassifier.scala
│   ├── OneVsRest.scala
│   └── ClassificationSummary.scala
│
├── regression/                  # 回归算法
├── tree/                        # 树算法公共组件
├── clustering/                  # 聚类算法
├── feature/                     # 特征工程 Transformers
│   ├── VectorAssembler.scala         # 特征拼接
│   ├── StandardScaler.scala          # 特征标准化
│   ├── StringIndexer.scala           # 类别编码
│   ├── OneHotEncoder.scala           # 独热编码
│   ├── HashingTF.scala               # 文本 Hash 特征
│   ├── IDF.scala
│   ├── PCA.scala
│   ├── Normalizer.scala
│   ├── ChiSqSelector.scala           # 特征选择
│   ├── Instance.scala                # Instance / InstanceBlock
│   └── ...
│
├── optim/                       # 优化算法核心
│   ├── aggregator/                    # 梯度/损失聚合器
│   │   ├── DifferentiableLossAggregator.scala
│   │   ├── BinaryLogisticBlockAggregator.scala   ★
│   │   ├── MultinomialLogisticBlockAggregator.scala ★
│   │   ├── LeastSquaresBlockAggregator.scala
│   │   ├── HingeBlockAggregator.scala
│   │   └── HuberBlockAggregator.scala
│   ├── loss/                          # 损失函数 + 正则化
│   │   ├── RDDLossFunction.scala
│   │   └── DifferentiableRegularization.scala
│   ├── NormalEquationSolver.scala
│   └── WeightedLeastSquares.scala
│
├── param/                       # 参数系统
├── linalg/                      # 线性代数 (Vector, Matrix, BLAS)
├── tuning/                      # 超参数调优 (CrossValidator, TrainValidationSplit)
├── evaluation/                  # 评估器
├── util/                        # 工具类 (MLWriter, MLReader, Instrumentation)
└── stat/                        # 统计工具 (Correlation, KolmogorovSmirnovTest, Summarizer)
```

## 2. 核心抽象层次

Spark ML 的架构遵循一个严格的类型层次，确保了极高的代码复用性：

```
PipelineStage (抽象基类)
├── Estimator[M <: Model[M]]           ← 可训练的东西（输入 Data → 输出 Model）
│   └── Predictor[FeaturesType, E, M]  ← 预测问题公共逻辑（schema 校验、label 转 Double）
│       └── Classifier[F, E, M]        ← 分类问题公共逻辑（rawPredictionCol, numClasses）
│           └── ProbabilisticClassifier[F, E, M]  ← 可输出概率的分类器
│               └── LogisticRegression  ← 具体实现
│
└── Transformer                        ← 可转换数据的东西（输入 Data → 输出 Data）
    ├── Model[M]                       ← Estimator.fit() 产出的模型
    │   └── PredictionModel[F, M]
    │       └── ClassificationModel[F, M]
    │           └── ProbabilisticClassificationModel[F, M]
    │               └── LogisticRegressionModel  ← 具体实现
    │
    └── (Feature Transformers)          ← VectorAssembler, StandardScaler, etc.
```

### 2.1 Estimator → Model 模式

这是 Spark ML 最核心的设计模式：

```scala
// Estimator 是工厂，fit() 产出 Model
abstract class Estimator[M <: Model[M]] extends PipelineStage {
  def fit(dataset: Dataset[_]): M
}

// Model 是 Transformer，transform() 产出带预测结果的 DataFrame
abstract class Model[M <: Model[M]] extends Transformer {
  def transform(dataset: Dataset[_]): DataFrame
}
```

**为什么这样设计？** 它统一了所有 ML 算法的使用接口。用户只需记住两个方法：`fit()` 训练、`transform()` 预测，不管是什么算法。同时 Pipeline 可以无差别地串联 Estimator 和 Transformer。

### 2.2 Predictor 的 schema 自动化

`Predictor` 基类做了大量脏活累活：

```scala
override def fit(dataset: Dataset[_]): M = {
  // 1. Schema 校验（features 必须是 Vector 类型，label 必须是数值）
  transformSchema(dataset.schema, logging = true)

  // 2. 自动类型转换（label 转 DoubleType, weight 转 DoubleType）
  val labelCasted = dataset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)

  // 3. 委派给子类实现具体训练逻辑
  copyValues(train(casted).setParent(this))
}
```

子类（如 `LogisticRegression`）只需要实现 `train(dataset: Dataset[_]): M`，不需要管 schema 校验、类型转换、参数拷贝等 boilerplate。

## 3. Pipeline 编排机制

Pipeline 是 Spark ML 的"胶水"，将多个 Estimator/Transformer 串成一个可复用的工作流：

```scala
// 典型的 ML Pipeline
val pipeline = new Pipeline().setStages(Array(
  new StringIndexer().setInputCol("category").setOutputCol("categoryIndex"),  // Estimator
  new OneHotEncoder().setInputCol("categoryIndex").setOutputCol("categoryVec"), // Transformer
  new VectorAssembler().setInputCols(Array("categoryVec", "age")).setOutputCol("features"), // Transformer
  new LogisticRegression() // Estimator（最后一个）
))

// fit: 按顺序执行，Estimator.fit → Model, Transformer 直接传递
val model = pipeline.fit(trainingData)

// transform: 按顺序执行 transform
val predictions = model.transform(testData)
```

**Pipeline.fit() 的核心逻辑**（[Pipeline.scala:133](mllib/src/main/scala/org/apache/spark/ml/Pipeline.scala#L133)）：

```scala
override def fit(dataset: Dataset[_]): PipelineModel = {
  transformSchema(dataset.schema, logging = true)
  var curDataset = dataset
  val transformers = ListBuffer.empty[Transformer]
  theStages.iterator.zipWithIndex.foreach { case (stage, index) =>
    if (index <= indexOfLastEstimator) {
      val transformer = stage match {
        case estimator: Estimator[_] => estimator.fit(curDataset)  // 训练出 Model
        case t: Transformer => t                                    // 直接使用
      }
      if (index < indexOfLastEstimator) {
        curDataset = transformer.transform(curDataset)  // 传给下一阶段
      }
      transformers += transformer
    }
  }
  new PipelineModel(uid, transformers.toArray)
}
```

关键点：
- 找到最后一个 Estimator 的位置
- 在最后一个 Estimator **之前**的每个阶段，fit 后立刻 transform 产生下一阶段输入
- 最后一个 Estimator 之后（通常没有数据需要再 transform），直接保存 transformer

## 4. 参数系统

Spark ML 使用了一套自研的参数系统（`org.apache.spark.ml.param`），所有参数都是强类型的：

```scala
// 参数定义（在 LogisticRegressionParams trait 中）
val regParam: DoubleParam = new DoubleParam(this, "regParam", "regularization parameter",
  ParamValidators.gtEq(0))

val family: Param[String] = new Param(this, "family",
  "The name of family which is a description of the label distribution to be used in the " +
    s"model. Supported options: ${supportedFamilyNames.mkString(", ")}.",
  (value: String) => supportedFamilyNames.contains(value.toLowerCase(Locale.ROOT)))

// Setter（流畅接口）
def setRegParam(value: Double): this.type = set(regParam, value)
def setFamily(value: String): this.type = set(family, value)

// Getter
def getRegParam: Double = $(regParam)  // $ 操作符取值，有默认值则返回
```

参数系统的关键特性：
- **验证器**：`ParamValidators.gtEq(0)` 自动校验
- **默认值**：`setDefault(regParam -> 0.0, maxIter -> 100)`
- **参数映射**：`ParamMap` 支持训练时覆盖 (`fit(dataset, paramMap)`)
- **参数拷贝**：`copyValues(model, extra)` 在 fit 完成后将参数复制到 Model

## 5. LogisticRegression 在架构中的位置

```
                    ┌──────────────────────┐
                    │     PipelineStage     │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼───────┐  ┌─────▼─────┐  ┌──────▼──────┐
     │   Estimator     │  │Transformer│  │  Params...  │
     └────────┬────────┘  └───────────┘  └─────────────┘
              │
     ┌────────▼───────────────────────────┐
     │        Predictor[Vector, E, M]     │
     │  - schema validation               │
     │  - label casting to DoubleType     │
     │  - weight casting                  │
     │  - train() abstract method         │
     └────────┬───────────────────────────┘
              │
     ┌────────▼───────────────────────────┐
     │     Classifier[Vector, E, M]       │
     │  - rawPredictionCol               │
     │  - numClasses inference            │
     │  - predictRaw() abstract           │
     └────────┬───────────────────────────┘
              │
     ┌────────▼───────────────────────────┐
     │ ProbabilisticClassifier[E, M]      │
     │  - probabilityCol                  │
     │  - raw2probabilityInPlace() abstract│
     │  - thresholds                      │
     └────────┬───────────────────────────┘
              │
     ┌────────▼───────────────────────────┐
     │  LogisticRegression + Params trait │  ← 具体实现
     │  - train() implementation           │
     │  - LBFGS / OWLQN / LBFGSB optimizer│
     │  - block aggregation               │
     └────────────────────────────────────┘
```

Model 侧对称：

```
                    ┌──────────────────────┐
                    │     Transformer       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │     Model[M]         │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
     ┌────────▼────────┐               ┌────────▼────────┐
     │PredictionModel  │               │  (other models) │
     └────────┬────────┘               └─────────────────┘
              │
     ┌────────▼───────────────────────────┐
     │ ClassificationModel                │
     │  - predictRaw()                    │
     │  - raw2prediction()                │
     └────────┬───────────────────────────┘
              │
     ┌────────▼───────────────────────────┐
     │ ProbabilisticClassificationModel   │
     │  - raw2probabilityInPlace()        │
     │  - probability2prediction()        │
     └────────┬───────────────────────────┘
              │
     ┌────────▼───────────────────────────┐
     │  LogisticRegressionModel           │  ← 具体实现
     │  - coefficientMatrix               │
     │  - interceptVector                 │
     │  - margins() / raw2probInPlace()   │
     └────────────────────────────────────┘
```

## 6. 关键设计决策解读

### 6.1 为什么有两套 ML API（ml vs mllib）？

- **`mllib`（旧）**：基于 RDD，API 直接暴露 `RDD[LabeledPoint]`，需要手动处理特征工程
- **`ml`（新）**：基于 DataFrame，自动处理 schema、类型转换、元数据；天然支持 Pipeline；与 Spark SQL 无缝集成

从 Spark 2.0 开始，`ml` 是推荐 API，`mllib` 进入维护模式。

### 6.2 为什么用 `Predictor` 中间层？

回归和分类有大量公共逻辑：schema 校验、label 类型转换、weight 列处理、predictionCol 管理。如果不提取 `Predictor`，这些代码会在 `Regressor` 和 `Classifier` 中各写一遍。

### 6.3 为什么 Model 也是 Transformer？

这样 `PipelineModel` 可以统一对待所有阶段——不管是特征转换器还是训练出的模型，都是 `Transformer.transform()`。这也意味着模型可以直接在 Pipeline 中使用。

## 7. 类大小统计

| 文件 | 行数 | 职责 |
|------|------|------|
| LogisticRegression.scala | 1585 | 训练器 + 模型 + Summary/SummaryImpl |
| LogisticRegressionSuite.scala | ~3000 | 测试覆盖 |
| BinaryLogisticBlockAggregator.scala | 155 | 二元逻辑回归梯度/损失聚合 |
| MultinomialLogisticBlockAggregator.scala | 222 | 多元逻辑回归梯度/损失聚合 |
| DifferentiableLossAggregator.scala | 82 | 聚合器基类 |
| RDDLossFunction.scala | 72 | Breeze DiffFunction 适配器 |
| DifferentiableRegularization.scala | 89 | L2 正则化实现 |
| Instance.scala | 212 | Instance / InstanceBlock 定义 |
