## `.raw` 文件处理流程说明

本文档说明 `assoG2P/bin/preprocess.py` 中对 PLINK `.raw` 基因型文件的**全部处理流程**，包括 `.raw` 的生成、解析、清洗、类型转换、分块/并行策略以及与表型数据的合并方式。

---

## 1. `.raw` 文件的来源与生成

- **生成入口函数**：`process_single_chromosome_optimized`
- **上游调用**：`plink_to_training_data_optimized` → `process_single_chromosome_optimized`

### 1.1 生成命令

对每条染色体，代码调用 PLINK 生成对应的 `.raw`：

- 基础命令（每条染色体）：
  - `--bfile {plink_prefix_abs}`
  - `--chr {chr_num}`
  - `--recodeA`
  - `--out {plink_prefix_abs}_chr{chr_num}`
- 其中：
  - `plink_prefix_abs`：全局的 PLINK 二进制前缀（已做样本/SNP过滤等预处理）
  - `chr_num`：当前处理的染色体编号
  - 输出 `.raw` 文件：`{plink_prefix_abs}_chr{chr_num}.raw`

### 1.2 关联的 `.bim` 文件与 SNP 映射

在解析 `.raw` 之前，会根据同一前缀的 `.bim` 文件构建 SNP 映射：

- 使用函数：`get_snp_chr_pos_mapping_optimized(bim_file, chr_num)`
- `.bim` 列结构：`chr snp_id genetic_dist physical_pos a1 a2`
- 生成的映射字典 `chr_snp_mapping` 主要规则：
  - `snp_id` → `"chr_physical_pos"`
  - 同时支持多种可能的 PLINK 列名格式：
    - `"chr:pos_a1"` → `"chr_pos"`
    - `"chr:pos_a2"` → `"chr_pos"`
    - `"chr:pos"`    → `"chr_pos"`
- 作用：在解析 `.raw` 时，将 SNP 列名统一为 `chr_pos` 形式，便于后续建模与合并。

---

## 2. `.raw` 文件的结构识别

### 2.1 表头解析

- 使用内存映射 `mmap` 打开 `.raw` 文件，只读取首行表头：
  - 找到第一行换行符，截取表头行并按空白切分为 `original_columns`。
- 典型 `.raw` 表头中的元信息列包括：
  - `FID`, `IID`, `PAT`, `MAT`, `SEX`, `PHENOTYPE`
- 代码逻辑：
  - **样本列**：找到列名为 `IID` 的列索引，作为样本 ID 列（后续重命名为 `sample`）
  - **SNP 列**：除上述元信息列外的所有列，都视为 SNP 基因型列

### 2.2 SNP 列名的规范化

在解析表头阶段，对每个 SNP 列名做两级处理：

1. **快速解析 PLINK 列名格式**（正则）：
   - 通过 `VectorizedProcessor.parse_plink_column_name` 解析形如：
     - `"chr:pos_allele"` → `("chr", "pos")`
   - 如果能解析成功，则初始映射为：`原列名` → `"chr_pos"`
2. **与 `.bim` 映射合并**：
   - 调用 `VectorizedProcessor.batch_parse_plink_columns(original_columns)` 得到一批规则化列名
   - 随后用 `chr_snp_mapping`（来自 `.bim`）覆盖/补充分布映射：
     - `.bim` 映射 **优先级更高**，保证 SNP 命名与 `.bim` 一致

最终得到一个 `batch_mapping` 字典，用于后续对 DataFrame 列名的批量重命名。

---

## 3. 分块读取与并行处理策略

核心解析函数：`parse_raw_file_optimized_v2`

### 3.1 缓存机制

- 生成缓存键：
  - `cache_key = "raw_parse_{raw_file}_{hash(frozenset(chr_snp_mapping.items()))}"`
- 若开启 `use_cache=True`：
  - 会在全局 `GLOBAL_CACHE` 中尝试读取已解析结果
  - 若缓存存在且文件未被修改，直接返回解析好的 DataFrame，避免重复 I/O 与解析

### 3.2 基于 `mmap` 的高效 I/O

1. 使用 `mmap` 映射整个 `.raw` 文件到内存
2. 仅用前一部分解析表头
3. 对表头之后的二进制块切片，包裹成 `io.BytesIO`，交给 `pandas.read_csv` 分块读取数据区

这样做的好处：

- 避免多次磁盘读写
- 可以利用 `pandas.read_csv` 的高性能文本解析能力

### 3.3 动态并行参数与分块大小

- **进程数 `max_workers`**：
  - 若未指定：
    - 根据 CPU 核数与文件大小动态估计
    - 大文件（> 1GB）时最多使用 8 个进程
    - 否则使用 `CPU 核数 / 2`，但不超过 4
- **行分块大小 `chunk_rows`**：
  - 下限：5000 行
  - 上限：50000 行
  - 目标：每个 chunk 的处理耗时约 1–2 秒，尽量让 CPU 保持高负载但不过度占用内存

### 3.4 第一阶段：串行分块读取

- 利用 `pandas.read_csv` 在 `mmap` 包裹的 `BytesIO` 上进行 **按行分块读取**：
  - `sep=r'\s+'`（空白分隔）
  - `header=None`，列名沿用头部解析出的 `original_columns`
  - `dtype=str`，先全部以字符串读入，便于统一处理缺失值和类型转换
  - `usecols`：仅读取样本列和 SNP 列
- 每个 chunk 保存为 `(chunk_idx, chunk_df)` 追加到 `raw_chunks` 列表中。

### 3.5 第二阶段：CPU 密集型并行处理

#### 3.5.1 多进程模式（默认）

- 条件：`use_parallel=True` 且 `max_workers > 1` 且 `chunk_count > 1`
- 为每个 chunk 构造参数元组：
  - `(chunk_idx, chunk_df, batch_mapping, sample_col_idx, original_columns, dtype_obj)`
- 使用 `ProcessPoolExecutor` 并行调用 `_process_chunk_parallel`：
  - 得到 `(chunk_idx, processed_chunk_df)` 并收集
  - 完成后按 `chunk_idx` 排序，保持与原文件的行顺序一致

#### 3.5.2 串行模式（兼容/小数据）

- 条件：不满足并行模式时（小文件或显式关闭并行）
- 顺序遍历 `raw_chunks`，逐个调用 `_process_chunk_parallel`，将结果依次追加到 `chunks` 列表中。

---

## 4. 单个分块 `_process_chunk_parallel` 的具体处理

辅助函数：`_process_chunk_parallel`

对每个 `chunk_df` 执行以下步骤：

1. **批量重命名 SNP 列**：
   - 调用 `VectorizedProcessor.batch_rename_columns(chunk_df, batch_mapping)`
   - 使用前文构建的 `batch_mapping` 将 SNP 列名统一为 `"chr_pos"` 形式
2. **样本列标准化**：
   - 使用最初识别的 `sample_col_idx` 找到样本列原始列名，例如 `IID`
   - 将该列重命名为 `sample`
3. **向量化类型转换**：
   - 调用 `VectorizedProcessor.vectorized_type_conversion`：
     - 排除列：`["sample"]`
     - 目标类型：`dtype_obj`（默认 `np.int8`）
   - 内部逻辑：
     - 优先尝试直接 `astype(target_dtype)`（数值型最优路径）
     - 若为字符串或混合类型：
       - 先尝试 `astype(float)` 再转 `target_dtype`
       - 若失败，使用 `pd.to_numeric(..., errors='coerce')` 转换为数值
       - 将 NaN 替换为 -1（缺失值编为 -1）
     - 对超大矩阵（元素>5000万）会再次按元素数切分为更小块进行转换，并可在线程级别进一步并行（注意：在进程并行模式下内部线程并行被关闭以避免过度并行）
4. **设置索引**：
   - 确保名为 `sample` 的列存在
   - 将 `sample` 设为行索引：`chunk_df.set_index("sample")`

返回值：`(chunk_idx, 处理后的 chunk_df)`。

---

## 5. 所有分块的合并与内存优化

### 5.1 分块结果合并

在所有 chunk 处理完成后：

- 如果只有一个 chunk：
  - 直接将该 DataFrame 作为最终结果 `result_df`
- 如果有多个 chunk：
  - 采用 **批次合并** 策略，减少一次性 `concat` 带来的内存峰值：
    - 设置批大小 `batch_size`：
      - 并行模式下：`max(10, max_workers * 2)`
      - 串行模式下：10
    - 每批将多个 chunk 沿行方向拼接（`axis=0`）
    - 各批次结果再进行二次合并得到最终 `result_df`

### 5.2 DataFrame 内存优化

所有 chunk 合并完成后，对整体 `result_df` 调用：

- `VectorizedProcessor.optimize_dataframe_memory(result_df)`：
  - 若某列唯一值比例 < 设定阈值（默认 0.5），可降为 `category` 类型
  - 对整数列根据数据范围降型（如 `int32` → `int8/uint8` 等）
  - 目标：在不损失信息的前提下最大程度降低内存占用

### 5.3 结果写入缓存

- 若开启 `use_cache=True`：
  - 将最终 `result_df` 写入 `GLOBAL_CACHE`，并关联原始 `.raw` 文件的修改时间
  - 若源文件未变，后续再次解析同一个 `.raw` 会直接复用缓存结果

---

## 6. 与表型的合并与下游使用

### 6.1 单条染色体层面的合并

解析完某条染色体的 `.raw` 后，`process_single_chromosome_optimized` 会执行：

1. 将表型 DataFrame `pheno_df` 统一以 `sample` 为索引
2. 调用：
   - `chr_geno_df.merge(pheno_df, left_index=True, right_index=True, how="inner")`
3. 得到包含：
   - 行：样本（`sample`）
   - 列：当前染色体所有 SNP（已统一为 `"chr_pos"` 命名）+ 表型列 `phenotype`

随后该染色体的结果会被保存为临时文件（优先 Feather，回退到 Pickle），供最终汇总使用。

### 6.2 全基因组层面的训练数据生成

在 `plink_to_training_data_optimized` 中：

1. 按染色体循环调用 `process_single_chromosome_optimized`，获得每条染色体的中间结果文件
2. 分批读回这些中间文件，并按列拼接（`concat(axis=1)`），仅保留各批次共同的样本索引
3. 最终得到：
   - 行：所有有效样本
   - 列：所有保留下来的 SNP（按 `chr_pos` 命名）+ `phenotype` 列
4. 将最终训练矩阵写出为 `.txt` 或 `.txt.gz`

---

## 7. 缺失值与编码约定

在 `.raw` 文件处理与类型转换过程中，采用以下约定：

- 原始 `.raw` 中的缺失基因型（如空值或无法解析的值）：
  - 在 `pd.to_numeric(..., errors='coerce')` 阶段会变为 `NaN`
  - 随后通过 `np.nan_to_num(..., nan=-1)` 或 `fillna(-1)` 被统一编码为 `-1`
- 代码中最终 SNP 类型通常为 `np.int8`：
  - 常见编码：
    - 0, 1, 2：实际等位基因剂量（PLINK recodeA 输出）
    - -1：缺失值

这一约定在下游训练和分析时需要保持一致，并且在模型侧要显式处理 `-1` 缺失编码。

---

## 8. 性能与资源控制要点

- **I/O 优化**：
  - 使用 `mmap` + `pandas.read_csv` 进行一次性顺序扫描，避免多次打开/关闭文件
- **CPU 利用**：
  - 大文件：利用多进程对 chunk 并行处理
  - 巨型数组：在单个 chunk 内部进一步按元素数量分片，并可在列级使用线程并行
- **内存控制**：
  - 读取时按行分块
  - 类型转换前后均进行降型和 `category` 压缩
  - 合并时采用“批次合并”避免一次性 `concat` 造成内存峰值
- **缓存利用**：
  - 对重复使用的 `.raw` 文件结果进行内存缓存，充分利用重复调用场景

---

## 9. 开发与排查建议

- 若需要确认 `.raw` 处理是否正确，可重点查看：
  - 列名是否已经统一为 `"chr_pos"` 形式
  - SNP 矩阵的值域是否仅为 `{0, 1, 2, -1}`
  - 行索引是否为样本 ID，且与表型文件中的样本匹配
- 若遇到内存不足：
  - 可以尝试：
    - 减小 `chunk_rows`
    - 关闭并行（`use_parallel=False`）以减少进程间内存复制
    - 或在上游进一步加严 SNP 过滤（`maf/geno` 参数）

