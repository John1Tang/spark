# LogisticRegression 训练深度解读

> 本文档从源码级别深入分析 Spark ML 中 LogisticRegression 的训练全流程，包括数据准备、优化算法、Block 化聚合、正则化策略，以及为什么这些设计能高效工作。

## 1. 训练入口：从 fit() 到 train()

### 1.1 调用链

```
LogisticRegression.fit(dataset)
  └→ Predictor.fit(dataset)                     // schema 校验 + label 类型转换
       └→ LogisticRegression.train(dataset)     // ★ 真正的训练逻辑
            └→ trainImpl(...)                   // 核心优化循环
```

### 1.2 train() 方法拆解

`LogisticRegression.train()` 在 [LogisticRegression.scala:506](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L506) 中实现，是整个训练流程的**编排器**。它不做数值计算，而是：

1. 提取 Instance RDD
2. 计算数据统计量（均值、标准差、类别分布）
3. 选择优化器（LBFGS / OWLQN / LBFGSB）
4. 构建初始解
5. 调用 `trainImpl()` 执行迭代优化
6. 后处理系数（反标准化、中心化）
7. 创建 Model 对象

```scala
protected[spark] def train(dataset: Dataset[_]): LogisticRegressionModel = instrumented { instr =>
  // Step 1: 提取 (label, weight, features) 三元组
  val instances = dataset.select(
    checkClassificationLabels($(labelCol), None),
    checkNonNegativeWeights(get(weightCol)),
    checkNonNanVectors($(featuresCol))
  ).rdd.map { case Row(l: Double, w: Double, v: Vector) =>
    Instance(l, w, v)
  }.setName("training instances")

  // Step 2: 计算数据统计量
  val (summarizer, labelSummarizer) = Summarizer
    .getClassificationSummarizers(instances, $(aggregationDepth), Seq("mean", "std", "count"))

  val numFeatures = summarizer.mean.size
  val histogram = labelSummarizer.histogram
  val numClasses = ...  // 从 metadata 或数据推断

  // Step 3: 边界情况处理（所有 label 相同 → 直接返回零模型）
  if ($(fitIntercept) && isConstantLabel && !usingBoundConstrainedOptimization) {
    return createModel(dataset, numClasses, zeroCoefMatrix, interceptVec, Array(0.0))
  }

  // Step 4: 构建正则化对象
  val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)
  val regularization = if (regParamL2 != 0.0) {
    Some(new L2Regularization(regParamL2, shouldApply, ...))
  } else None

  // Step 5: 选择优化器
  val optimizer = createOptimizer(numClasses, numFeatures, featuresStd, ...)

  // Step 6: 初始解
  val initialSolution = createInitialSolution(...)

  // Step 7: 执行优化
  val (allCoefficients, objectiveHistory) = trainImpl(instances, ..., initialSolution, ...)

  // Step 8: 后处理 - 反标准化系数
  // ... 将 scaled-space 系数转换回 original-space

  // Step 9: 创建 Model
  return createModel(dataset, numClasses, coefficientMatrix, interceptVector, objectiveHistory)
}
```

## 2. 数据流：从 DataFrame 到 InstanceBlock

### 2.1 Instance 数据结构

[Instance.scala:33](mllib/src/main/scala/org/apache/spark/ml/feature/Instance.scala#L33):

```scala
case class Instance(label: Double, weight: Double, features: Vector)
```

这是 Spark ML 线性算法的**统一数据表示**。所有的线性模型（LR、LinearSVC、LinearRegression、AFTSurvivalRegression）都用这个结构。

### 2.2 Block 化：性能关键优化

从 Spark 3.1.0 开始，LR 训练引入了 **InstanceBlock** 机制，这是性能提升的关键：

```scala
// 传统方式：逐条处理 Instance
RDD[Instance] → 逐条计算 gradient

// Block 方式：批量处理
RDD[Instance] → blokifyWithMaxMemUsage() → RDD[InstanceBlock] → 批量计算
```

[Instance.scala:147](mllib/src/main/scala/org/apache/spark/ml/feature/Instance.scala#L147):

```scala
def blokifyWithMaxMemUsage(instanceIterator: Iterator[Instance], maxMemUsage: Long): Iterator[InstanceBlock] = {
  // 将多个 Instance 打包成一个 InstanceBlock，内存不超过 maxMemUsage（默认 1MB）
  // 这样可以用 GEMV/GEMM 批量计算，而不是逐条循环
}
```

**为什么 Block 化能加速？**

1. **BLAS Level-2/3 优化**：逐条处理调用 `y = α*A*x + β*y`（GEMV）一次算一个样本；Block 化后可以用矩阵-矩阵乘法 `C = α*A*B + β*C`（GEMM），充分利用 CPU SIMD 指令和缓存
2. **减少函数调用开销**：一个 block 可能包含几十到几百个样本，一次 `add(block)` 替代几十次 `add(instance)`
3. **缓存友好**：block 内的特征矩阵连续存储，CPU 缓存命中率更高

### 2.3 InstanceBlock 结构

```scala
case class InstanceBlock(
    labels: Array[Double],    // 多个样本的 label
    weights: Array[Double],   // 多个样本的 weight（全为 1 时为空数组优化）
    matrix: Matrix            // 特征矩阵，transposed=true（行=样本，列=特征）
)
```

关键约束：`matrix.isTransposed == true`，即存储格式是 `numSamples × numFeatures`。这是为了和 BLAS GEMV/GEMM 的输入格式匹配。

## 3. 标准化策略

### 3.1 训练空间 vs 原始空间

Spark ML 的 LR 实现有一个重要设计：**训练在标准化空间中进行，但返回的系数在原始空间**。

```scala
// trainImpl() 中：
val inverseStd = featuresStd.map(std => if (std != 0) 1.0 / std else 0.0)
val scaledMean = Array.tabulate(numFeatures)(i => inverseStd(i) * featuresMean(i))

// 标准化数据：x' = x / std
val scaled = instances.mapPartitions { iter =>
  val func = StandardScalerModel.getTransformFunc(Array.empty, bcInverseStd.value, false, true)
  iter.map { case Instance(label, weight, vec) => Instance(label, weight, func(vec)) }
}
```

**为什么这样做？**

1. **收敛速度**：标准化后所有特征在同一尺度，梯度下降收敛更快（条件数更小）
2. **正则化公平性**：L2 正则化 `||w||^2` 在标准化空间中对所有特征公平
3. **用户透明**：返回的系数是原始空间的，用户不需要手动转换

### 3.2 系数反标准化

训练完成后，需要将系数从标准化空间转回原始空间：

```scala
// w_original = w_scaled / std
allCoefMatrix.foreachActive { (classIndex, featureIndex, value) =>
  if (featuresStd(featureIndex) != 0.0) {
    denseCoefficientMatrix.update(classIndex, featureIndex,
      value / featuresStd(featureIndex))
  }
  // 截距项不需要缩放（因为 y 没有标准化）
  if (isIntercept) interceptVec.toArray(classIndex) = value
}
```

**数学验证：**

```
标准化空间：y = sigmoid(w_scaled * x' + b) = sigmoid(w_scaled * (x/std) + b)
原始空间：  y = sigmoid(w_original * x + b)

对比得：w_original = w_scaled / std ✓
```

### 3.3 中心化优化（fitWithMean）

当 `fitIntercept=true` 且没有对截距的 bound 约束时，Spark 会做**虚拟中心化**：

```scala
val fitWithMean = $(fitIntercept) &&
  (!isSet(lowerBoundsOnIntercepts) || ...) &&
  (!isSet(upperBoundsOnIntercepts) || ...)
```

这不是真的减去均值，而是在计算时通过调整截距来实现等价效果。训练后需要**反调整**：

```scala
// 训练前：调整初始解的截距
val adapt = BLAS.ddot(numFeatures, initialSolution, 1, scaledMean, 1)
initialSolution(numFeatures) += adapt

// 训练后：反向调整最终解的截距
val adapt = BLAS.ddot(numFeatures, solution, 1, scaledMean, 1)
solution(numFeatures) -= adapt
```

## 4. 优化算法选择

### 4.1 三种优化器的选择逻辑

[LogisticRegression.scala:770-809](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L770-L809):

```scala
private def createOptimizer(...): FirstOrderMinimizer[...] = {
  val regParamL1 = $(elasticNetParam) * $(regParam)

  if ($(elasticNetParam) == 0.0 || $(regParam) == 0.0) {
    // 纯 L2 或无正则化
    if (lowerBounds != null && upperBounds != null) {
      // 有界约束优化 → LBFGSB
      new BreezeLBFGSB(lowerBounds, upperBounds, $(maxIter), 10, $(tol))
    } else {
      // 无约束优化 → LBFGS
      new BreezeLBFGS($(maxIter), 10, $(tol))
    }
  } else {
    // ElasticNet (L1 + L2) → OWLQN
    // OWLQN 专门处理 L1 正则化的不可微问题
    val regParamL1Fun = (index: Int) => {
      // 截距不加 L1 惩罚
      if (isIntercept) 0.0
      else if (standardization) regParamL1
      else regParamL1 / featuresStd(featureIndex)  // 反标准化
    }
    new BreezeOWLQN($(maxIter), 10, regParamL1Fun, $(tol))
  }
}
```

### 4.2 优化器对比

| 优化器 | 适用场景 | 数学原理 | 收敛速度 |
|--------|---------|---------|---------|
| **LBFGS** | 无约束或纯 L2 | L-BFGS 拟牛顿法，用历史梯度近似 Hessian 逆 | 超线性 |
| **LBFGSB** | 有界约束 (box constraint) | LBFGS + 投影梯度，确保解在 [lower, upper] 内 | 超线性 |
| **OWLQN** | L1 或 ElasticNet | 正交方向 LBFGS，处理 L1 在零点不可微 | 超线性（在光滑子空间） |

**关键洞察**：LR 的损失函数是**凸的**（logistic loss 是凸函数），所以这些一阶优化器都能保证收敛到全局最优。

### 4.3 L1 正则化为什么需要 OWLQN？

标准 LBFGS 要求目标函数处处可微。但 L1 正则化 `λ * ||w||_1` 在 `w_i = 0` 处不可微（次梯度）。OWLQN（Orthant-Wise L-BFGS）的解决思路：

1. 将参数空间划分为 2^n 个"卦限"（orthant），每个卦限内 L1 是光滑的（符号固定）
2. 在每个卦限内用标准 L-BFGS
3. 如果迭代跨越卦限边界，重新计算

这就是为什么 ElasticNet 需要单独的 OWLQN 而不是简单修改 LBFGS。

## 5. 迭代优化核心：RDDLossFunction + BlockAggregator

### 5.1 Breeze 优化器接口

Breeze 的优化器需要一个 `DiffFunction[T]`：

```scala
trait DiffFunction[T] {
  def calculate(x: T): (Double, T)  // 返回 (loss, gradient)
}
```

`RDDLossFunction` 将这个接口桥接到 Spark RDD 计算：

[RDDLossFunction.scala:47](mllib/src/main/scala/org/apache/spark/ml/optim/loss/RDDLossFunction.scala#L47):

```scala
class RDDLossFunction[T, Agg <: DifferentiableLossAggregator[T, Agg]](
    instances: RDD[T],
    getAggregator: (Broadcast[Vector] => Agg),
    regularization: Option[DifferentiableRegularization[Vector]],
    aggregationDepth: Int = 2)
  extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    // 1. 广播当前系数到所有 executor
    val bcCoefficients = instances.context.broadcast(Vectors.fromBreeze(coefficients))

    // 2. 创建聚合器
    val thisAgg = getAggregator(bcCoefficients)

    // 3. treeAggregate: 在每个分区内聚合，然后树形合并
    val seqOp = (agg: Agg, x: T) => agg.add(x)        // 分区内
    val combOp = (agg1: Agg, agg2: Agg) => agg1.merge(agg2)  // 分区间
    val newAgg = instances.treeAggregate(thisAgg, seqOp, combOp, aggregationDepth, true)

    // 4. 加上正则化项
    val gradient = newAgg.gradient
    val regLoss = regularization.map { regFun =>
      val (regLoss, regGradient) = regFun.calculate(Vectors.fromBreeze(coefficients))
      BLAS.axpy(1.0, regGradient, gradient)  // gradient += regGradient
      regLoss
    }.getOrElse(0.0)

    bcCoefficients.destroy()
    (newAgg.loss + regLoss, gradient.asBreeze.toDenseVector)
  }
}
```

**关键设计**：
- `treeAggregate` 而不是 `aggregate`：使用树形归并减少 driver 压力，深度参数 `aggregationDepth`（默认 2）控制树的高度
- 广播系数：避免每次 task 序列化传输系数
- 聚合器模式：将 loss 和 gradient 的计算统一为一个可合并的 monoid

### 5.2 迭代循环

```scala
val states = optimizer.iterations(new CachedDiffFunction(costFun),
  new BDV[Double](initialSolution))

var state: optimizer.State = null
while (states.hasNext) {
  state = states.next()
  arrayBuilder += state.adjustedValue  // 记录每次迭代的目标函数值
}
```

`optimizer.iterations()` 返回一个 `Iterator[State]`，每次 `.next()` 触发一次：
1. 调用 `costFun.calculate(currentCoefficients)` → 触发全量 `treeAggregate`
2. 根据 loss 和 gradient 更新系数
3. 检查收敛条件（gradient 范数 < tol 或达到 maxIter）

## 6. BlockAggregator：梯度计算的核心

### 6.1 DifferentiableLossAggregator 基类

[DifferentiableLossAggregator.scala:30](mllib/src/main/scala/org/apache/spark/ml/optim/aggregator/DifferentiableLossAggregator.scala#L30):

```scala
trait DifferentiableLossAggregator[Datum, Agg] extends Serializable {
  protected var weightSum: Double = 0.0
  protected var lossSum: Double = 0.0
  protected val dim: Int
  protected lazy val gradientSumArray: Array[Double] = Array.ofDim[Double](dim)

  def add(instance: Datum): Agg  // 子类实现

  def merge(other: Agg): Agg = {  // 聚合器合并（用于 treeAggregate 的 combOp）
    weightSum += other.weightSum
    lossSum += other.lossSum
    BLAS.axpy(1.0, other.gradientSumArray, this.gradientSumArray)  // gradient += other.gradient
    this
  }

  def gradient: Vector = {
    val result = Vectors.dense(gradientSumArray.clone())
    BLAS.scal(1.0 / weightSum, result)  // 除以总权重得到平均梯度
    result
  }

  def loss: Double = lossSum / weightSum  // 平均损失
}
```

**关键设计**：聚合器是一个 **Monoid**（有 zero element 和 associative merge），这使得它可以高效地并行化。

### 6.2 BinaryLogisticBlockAggregator

这是二元逻辑回归的梯度计算核心。让我们一步一步拆解：

[BinaryLogisticBlockAggregator.scala:83](mllib/src/main/scala/org/apache/spark/ml/optim/aggregator/BinaryLogisticBlockAggregator.scala#L83):

```scala
def add(block: InstanceBlock): this.type = {
  // === Phase 1: 计算所有样本的 margin（边界值） ===
  // margin = w · x + b
  val arr = buffer  // 复用缓冲区
  if (fitIntercept) {
    val offset = coefficientsArray.last  // 截距
    java.util.Arrays.fill(arr, 0, size, offset)
    BLAS.gemv(1.0, block.matrix, coefficientsArray, 1.0, arr)  // arr = X * w + b
  } else {
    BLAS.gemv(1.0, block.matrix, coefficientsArray, 0.0, arr)  // arr = X * w
  }

  // === Phase 2: margin → loss + multiplier ===
  var localLossSum = 0.0
  var multiplierSum = 0.0
  var i = 0
  while (i < size) {
    val weight = block.getWeight(i)
    if (weight > 0) {
      val label = block.getLabel(i)  // 0 或 1
      val margin = arr(i)

      // 数值稳定的 logistic loss 计算
      if (label > 0) {
        localLossSum += weight * Utils.log1pExp(-margin)  // = weight * log(1 + exp(-margin))
      } else {
        localLossSum += weight * (Utils.log1pExp(-margin) + margin)
      }

      // 计算梯度乘子：weight * (sigmoid(margin) - label)
      // = weight * (1/(1+exp(-margin)) - label)
      val multiplier = weight * (1.0 / (1.0 + math.exp(-margin)) - label)
      arr(i) = multiplier  // 原地覆写为 multiplier
      multiplierSum += multiplier
    }
    i += 1
  }

  // === Phase 3: 计算梯度 ===
  // 梯度 = X^T * multiplier
  BLAS.gemv(1.0, block.matrix.transpose, arr, 1.0, gradientSumArray)

  // 截距的梯度 = sum(multiplier)
  if (fitIntercept) {
    gradientSumArray(numFeatures) += multiplierSum
  }

  this
}
```

### 6.3 数学验证

二元逻辑回归：

```
P(y=1|x) = sigmoid(w·x + b) = 1 / (1 + exp(-(w·x+b)))

Loss (负对数似然) = -Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
                  = Σ [log(1+exp(-m_i))]    (对 y=1 样本)
                  = Σ [log(1+exp(-m_i)) + m_i]  (对 y=0 样本)

Gradient = Σ [(p_i - y_i) * x_i] = Σ [multiplier_i * x_i]
```

代码中的 `log1pExp(-margin)` 正是 `log(1 + exp(-margin))` 的数值稳定版本：

```scala
// [Utils.scala:91]
def log1pExp(x: Double): Double = {
  if (x > 0) {
    x + math.log1p(math.exp(-x))  // x > 0 时，用这个避免 exp(x) 溢出
  } else {
    math.log1p(math.exp(x))       // x ≤ 0 时，直接计算
  }
}
```

**为什么不用 `log(1 + exp(x))`？** 因为当 `x > 709.78` 时，`exp(x)` 溢出到 `Infinity`，而 `x + log1p(exp(-x))` 可以安全处理。

### 6.4 MultinomialLogisticBlockAggregator

多元逻辑回归的核心区别：

1. **Margin 计算**：用 GEMM（矩阵-矩阵乘法）代替 GEMV：

```scala
// margins = X * W^T + b, 结果形状: S × C (样本数 × 类别数)
BLAS.gemm(1.0, block.matrix, linear.transpose, 1.0, arr)
```

2. **Softmax 替代 Sigmoid**：

```scala
Utils.softmax(arr, numClasses, i, size, arr)  // 将 margins 转为概率
localLossSum -= weight * math.log(arr(labelIndex))  // cross-entropy loss
arr(labelIndex) -= weight  // multiplier for correct class
```

3. **梯度计算**：对稀疏矩阵有专门优化的 GEMM：

```scala
block.matrix match {
  case dm: DenseMatrix =>
    BLAS.nativeBLAS.dgemm("T", "T", numClasses, numFeatures, size, 1.0, ...)
  case sm: SparseMatrix =>
    // 手动实现稀疏 GEMM，避免 DenseMatrix 转换
    while (colCounterForB < nB) {
      while (colCounterForA < kA) {
        gradientSumArray(...) += Avals(i) * Bval
      }
    }
}
```

## 7. L2 正则化实现

[DifferentiableRegularization.scala:50](mllib/src/main/scala/org/apache/spark/ml/optim/loss/DifferentiableRegularization.scala#L50):

```scala
class L2Regularization(
    regParam: Double,
    shouldApply: Int => Boolean,        // 哪些系数需要正则化（截距除外）
    applyFeaturesStd: Option[Int => Double]  // 非标准化时的反标准化因子
) extends DifferentiableRegularization[Vector] {

  override def calculate(coefficients: Vector): (Double, Vector) = {
    var sum = 0.0
    val gradient = new Array[Double](dv.size)
    dv.values.indices.filter(shouldApply).foreach { j =>
      val coef = coefficients(j)
      applyFeaturesStd match {
        case Some(getStd) =>
          val std = getStd(j)
          if (std != 0.0) {
            val temp = coef / (std * std)
            sum += coef * temp
            gradient(j) = regParam * temp
          }
        case None =>
          sum += coef * coef
          gradient(j) = coef * regParam
      }
    }
    (0.5 * sum * regParam, Vectors.dense(gradient))
  }
}
```

**为什么有 `applyFeaturesStd` 分支？**

当 `standardization=false` 时，训练仍在标准化空间中进行（为了收敛速度），但正则化需要映射回原始空间。对第 j 个特征：

```
w_scaled = w_original * std_j

L2 惩罚 (scaled space): λ * w_scaled^2 = λ * (w_original * std_j)^2

我们想要: λ * w_original^2

所以: 梯度调整 = λ * w_scaled / std_j^2
```

## 8. 初始解策略

好的初始解可以显著减少迭代次数：

[LogisticRegression.scala:815-928](mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L815-L928):

```scala
private def createInitialSolution(...): DenseMatrix = {
  val initialCoefWithInterceptMatrix = DenseMatrix.zeros(...)

  if (initialModelIsValid) {
    // 用户提供初始模型 → 使用其系数（需缩放）
    providedCoef.foreachActive { (classIndex, featureIndex, value) =>
      initialCoefWithInterceptMatrix.update(classIndex, featureIndex,
        value * featuresStd(featureIndex))  // 转到 scaled space
    }
  } else if ($(fitIntercept) && isMultinomial) {
    // 多元：截距初始化为 label 分布的对数
    val rawIntercepts = histogram.map(math.log1p)  // log(count_k + 1)
    val rawMean = rawIntercepts.sum / rawIntercepts.length
    rawIntercepts.indices.foreach { i =>
      initialCoefWithInterceptMatrix.update(i, numFeatures, rawIntercepts(i) - rawMean)
    }
  } else if ($(fitIntercept)) {
    // 二元：截距初始化为 log(count_1 / count_0)
    initialCoefWithInterceptMatrix.update(0, numFeatures,
      math.log(histogram(1) / histogram(0)))
  }
}
```

**数学直觉**：如果所有系数为零，`sigmoid(b)` 应该等于正样本比例。解得 `b = log(P(1)/P(0)) = log(count_1/count_0)`。这是一个很好的起点。

## 9. 边界情况处理

### 9.1 所有 label 相同

```scala
if ($(fitIntercept) && isConstantLabel && !usingBoundConstrainedOptimization) {
  // 不需要训练！系数全为零，截距为 ±∞
  val constantLabelIndex = Vectors.dense(histogram).argmax
  val interceptVec = if (isMultinomial) {
    Vectors.sparse(numClasses, Seq((constantLabelIndex, Double.PositiveInfinity)))
  } else {
    Vectors.dense(if (numClasses == 2) Double.PositiveInfinity else Double.NegativeInfinity)
  }
  return createModel(dataset, numClasses, zeroCoefMatrix, interceptVec, Array(0.0))
}
```

这是**早期退出优化**。如果所有样本 label 相同，模型就是"永远预测这个类"。不需要任何优化迭代。

### 9.2 无截距 + 常数非零列

```scala
if (!$(fitIntercept) && (0 until numFeatures).exists { i =>
  featuresStd(i) == 0.0 && featuresMean(i) != 0.0 }) {
  logWarning("... outputs zero coefficients for constant nonzero columns ...")
}
```

这是已知行为警告（与 R glmnet 一致）。如果有一个特征值恒定非零且没有截距项，优化器会将其系数设为 0。

## 10. 性能优化总结

| 优化技术 | 位置 | 效果 |
|---------|------|------|
| **Block 化** | InstanceBlock | GEMV→GEMM，CPU 缓存优化 |
| **标准化训练** | trainImpl | 条件数减小，收敛更快 |
| **虚拟中心化** | fitWithMean | 进一步改善条件数 |
| **数值稳定 loss** | log1pExp | 避免 exp 溢出 |
| **树形聚合** | treeAggregate(aggregationDepth) | 减少 driver 压力 |
| **广播系数** | bcCoefficients | 减少网络传输 |
| **系数复用** | buffer 数组 | 避免重复分配 |
| **早期退出** | isConstantLabel | 跳过不必要的训练 |
| **智能初始化** | label 分布初始化截距 | 减少迭代次数 |
| **稀疏 GEMM** | MultinomialAggregator | 稀疏特征矩阵加速 |

## 11. 端到端训练流程图

```
                    DataFrame
                        │
                        ▼
              ┌─────────────────────┐
              │  Predictor.fit()    │
              │  - schema check     │
              │  - label cast       │
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  LR.train()         │
              │  1. Extract Instance│
              │  2. Compute stats   │
              │  3. Select optimizer│
              │  4. Build initial   │
              └────────┬────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │LBFGS     │ │OWLQN     │ │LBFGSB    │
    │(pure L2) │ │(Elastic) │ │(bounded) │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         └────────────┼────────────┘
                      │
                      ▼
              ┌─────────────────────┐
              │  RDDLossFunction    │
              │  calculate(w)       │
              │  ├─ treeAggregate   │
              │  │   ├─ BlockAgg    │
              │  │   │  ├─ GEMV/GEMV│
              │  │   │  └─ softmax  │
              │  │   └─ merge       │
              │  └─ + regularization│
              └────────┬────────────┘
                       │
                       ▼ (迭代收敛)
              ┌─────────────────────┐
              │  Post-processing    │
              │  - 反标准化系数      │
              │  - 多元中心化        │
              └────────┬────────────┘
                       │
                       ▼
              LogisticRegressionModel
```
