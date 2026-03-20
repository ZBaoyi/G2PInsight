# assoG2P Genomic Analysis Tool

## Table of Contents
- [Feature Overview](#feature-overview)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Result Visualization](#3-result-visualization)
- [Output Files](#output-files)
  - [1. Preprocess Output](#1-preprocess-output)
  - [2. Train Output](#2-train-output)
  - [3. Train-all Output](#3-train-all-output)
  - [4. Predict Output](#4-predict-output)
  - [5. Visualize Output](#5-visualize-output)
- [Sample Data](#sample-data)
- [FAQ](#faq)
- [Developer Guide](#developer-guide)
- [License](#license)

---

## Feature Overview ##
**End-to-end genomic analysis toolkit** designed for Genome-Wide Association Studies (GWAS) with three core modules:

### 1. Data Preprocessing
- **功能**：将VCF格式基因型数据与表型数据整合为机器学习可直接使用的格式
- **支持格式**：VCF v4.2+ (bgzip压缩或未压缩)
- **核心特性**：
  - 自动检测SNP位置和样本ID
  - 缺失值处理（支持删除或填充）
  - 基因型编码转换（0|0→0, 0|1/1|0→1, 1|1→2）
  - 大型数据集抽样支持（默认50万点上限）

### 2. Model Training
- **功能**：通过机器学习算法识别与表型相关的遗传变异
- **支持算法**：LightGBM、RandomForest、XGBoost、SVM、CatBoost、Logistic Regression
- **核心特性**：
  - 自动区分分类/回归任务
  - 内置5折交叉验证
  - 超参数自动优化（RandomizedSearchCV）
  - 特征重要性计算与排序
  - GWAS特征筛选
  - LD过滤
  - 综合特征筛选（GWAS + LD）

### 3. Result Visualization
- **功能**：生成全基因组特征重要性散点图
- **图表类型**：全基因组特征重要性散点图（Manhattan-style scatter plot）
- **核心特性**：
  - 静态（PNG）和交互式（HTML）输出
  - 自动计算显著性阈值
  - 染色体位置自动标注
  - 正负效应颜色区分
  - 支持自动列名检测
---

## Installation Guide ##
### System Requirements
- Python 3.8-3.12 (64-bit)  
  **重要：请在安装前确认当前环境的 Python 版本在 3.8 到 3.12 之间。**
  - 可用命令检查：
    - `python --version` 或 `python3 --version`
  - 如果版本不在此范围内，建议创建单独的 conda 环境，例如：
    - `conda create -n assoG2P-env python=3.10`
    - `conda activate assoG2P-env`
- Linux/macOS: Bash shell

### Quick Install

#### Linux/macOS
```bash
chmod +x tools.sh
./tools.sh
```

安装流程包括：
首先会创建conda环境（已存在则不再重复创建），然后在conda中创建bioenv环境，再在该环境下安装以下生物信息分析工具：
```text
    "plink"         # 全基因组关联分析
    "fastqc"        # 质控检查
    "bwa"           # 序列比对
    "samtools"      # SAM工具集
    "bcftools"      # VCF处理
    "bedtools"      # 基因组算术
    "blast"         # 序列比对
    "bowtie2"       # 短序列比对
```

进入该环境：
```bash
conda activate bioenv
```

然后可以使用PLINK等工具进行基因型数据的筛选、过滤、质量控制。

### Whl Package Installation
推荐使用whl包进行安装，这是最简单快捷的安装方式。

#### 下载whl包
whl包可以从以下位置获取：
- **项目仓库**：从GitHub仓库的`dist/`目录下载whl文件
  - 访问：`https://github.com/chenrf0407/G2O_tool/tree/master/dist/`
  - 下载文件：`assoG2P-1.0.0-py3-none-any.whl`
- **本地构建**：如果已克隆项目，whl文件位于项目根目录的`dist/`目录下
- **文件名格式**：`assoG2P-1.0.0-py3-none-any.whl`

#### 安装whl包
**步骤1：下载whl文件**
```bash
# 方法1：从GitHub仓库下载（需要手动下载）
# 访问 https://github.com/chenrf0407/G2O_tool/tree/master/dist/
# 下载 assoG2P-1.0.0-py3-none-any.whl 文件

# 方法2：如果已克隆项目，whl文件在dist目录下
cd /path/to/assoG2P
ls dist/  # 查看whl文件
```

**步骤2：安装whl包**
```bash
# 激活conda环境
conda activate bioenv

# 进入whl文件所在目录，或使用完整路径
cd /path/to/whl/file

# 安装whl包（推荐方式）
pip install assoG2P-1.0.0-py3-none-any.whl

# 也可以直接指定whl文件的完整路径
pip install /path/to/assoG2P-1.0.0-py3-none-any.whl
```

**安装成功提示**：
```bash
Installing collected packages: assoG2P
Successfully installed assoG2P-1.0.0
```

### Alternative Installation (Source Code)
如果whl包不可用，可以从源码安装：
```bash
# 确保已安装tools.sh中的依赖
pip install .
```

### 环境验证
安装完成后验证环境：
```bash
conda activate bioenv
assog2p --version
# 应显示 assoG2P 1.0.0
```

### 常见安装问题
**Q: 如何选择安装方式？**
- **推荐**：使用whl包安装（最简单快捷）
- **备选**：从源码安装（需要编译，可能较慢）

**Q: 安装时出现权限问题**
```bash
# 方法1：使用用户安装模式（推荐）
pip install --user assoG2P-1.0.0-py3-none-any.whl

# 方法2：Linux/macOS添加sudo前缀
sudo pip install assoG2P-1.0.0-py3-none-any.whl

# 方法3：在conda环境中安装（无需sudo）
conda activate bioenv
pip install assoG2P-1.0.0-py3-none-any.whl
```

**Q: 如何验证安装成功？**
安装成功后，会显示以下信息：
```bash
Installing collected packages: assoG2P
Successfully installed assoG2P-1.0.0
```

然后验证安装：
```bash
# 检查版本
assog2p --version
# 应显示：assoG2P 1.0.0

# 查看帮助信息
assog2p -h
```

**Q: whl包在哪里下载？**
- **注意**：本项目尚未发布到PyPI，无法通过`pip install assoG2P`直接安装
- 从项目GitHub仓库的`dist/`目录下载：`https://github.com/chenrf0407/G2O_tool/tree/master/dist/`
- 或从项目本地`dist/`目录获取（如果已克隆项目）
- whl文件名格式：`assoG2P-{version}-py3-none-any.whl`
- 下载后需要先保存到本地，然后使用`pip install`命令安装本地whl文件

**Q: 安装whl包时出现"Preparing metadata (pyproject.toml) ... error"或CMake版本错误**
这是依赖包（如xgboost）需要从源码编译，但CMake版本过低导致的。解决方法：

```bash
# 方法1：升级CMake（推荐）
# 在conda环境中升级CMake
conda activate bioenv
conda install cmake>=3.18 -c conda-forge

# 然后重新安装whl包
pip install assoG2P-1.0.0-py3-none-any.whl
```

```bash
# 方法2：使用预编译的依赖包（如果CMake无法升级）
# 先单独安装预编译的xgboost和其他依赖
conda activate bioenv
pip install --prefer-binary xgboost lightgbm catboost

# 然后安装whl包（跳过已安装的依赖）
pip install --no-deps assoG2P-1.0.0-py3-none-any.whl

# 最后安装其他缺失的依赖（如果有）
pip install numpy pandas scikit-learn matplotlib plotly kaleido seaborn shap numba
```

```bash
# 方法3：使用conda安装依赖（推荐在conda环境中使用）
conda activate bioenv
conda install -c conda-forge xgboost lightgbm catboost numpy pandas scikit-learn matplotlib plotly seaborn

# 然后安装whl包（跳过依赖安装）
pip install --no-deps assoG2P-1.0.0-py3-none-any.whl

# 安装其他pip-only的依赖
pip install kaleido shap numba
```

**错误信息示例**：
```
CMake Error: CMake 3.18 or higher is required. You are running version 3.16.9
```
如果看到类似错误，请使用上述方法之一解决。

```bash
# 方法4：清除pip缓存后重试（如果上述方法都失败）
# 有时pip缓存中的损坏文件会导致编译问题
conda activate bioenv
pip cache purge  # 清除所有pip缓存
pip install assoG2P-1.0.0-py3-none-any.whl
```

```bash
# 方法5：创建新的conda环境重新安装（推荐用于解决复杂编译问题）
# 确保使用符合要求的Python版本（3.8-3.12）
conda create -n assoG2P-new python=3.10  # 使用3.10作为示例，可根据需要选择3.8-3.12
conda activate assoG2P-new

# 安装CMake和构建工具
conda install cmake>=3.18 -c conda-forge
conda install -c conda-forge xgboost lightgbm catboost numpy pandas scikit-learn matplotlib plotly seaborn

# 安装whl包
pip install assoG2P-1.0.0-py3-none-any.whl

# 安装其他pip-only的依赖
pip install kaleido shap numba

# 验证安装
assog2p --version
```

**如果遇到编译器相关错误（如GCC版本不兼容、编译超时等）**：
1. 首先尝试清除pip缓存：`pip cache purge`
2. 如果问题持续，建议创建新的conda环境，使用符合要求的Python版本（3.8-3.12）重新安装
3. 新环境可以避免旧环境的依赖冲突和缓存问题

**Q: 安装后找不到`assog2p`命令**
```bash
# 确保已激活conda环境
conda activate bioenv

# 检查Python路径
which python
which pip

# 如果使用--user安装，确保PATH包含用户site-packages目录
# Linux: ~/.local/bin
# macOS: ~/Library/Python/{version}/bin
```

安装完成后，`assog2p` 命令会在当前环境中注册。运行 `assog2p -h` 查看使用帮助。

---

## Quick Start ##
**Complete Workflow Example**

```bash
# Step 1: Data preprocessing (输出到目录，会生成train_data.txt)
assog2p preprocess -g genotype.vcf -p phenotype.txt -o preprocessed/

# Step 2: Model training (使用preprocess输出目录下的metadata.json)
assog2p train -j preprocessed/train_data_metadata.json -m LightGBM -o results/

# Step 3: Visualization
# Feature importance visualization
assog2p visualize -i results/LightGBM/LightGBM_feature_importance.txt -o result

# Model evaluation visualization (from plotting_data)
assog2p visualize -I results/LightGBM/LightGBM_plotting_data.npz -o result
```

**Advanced Example with GWAS and LD Filtering**:
```bash
# Preprocessing (输出到目录，会生成train_data.txt和train_data_metadata.json)
assog2p preprocess -g genotype.vcf -p phenotype.txt -o preprocessed/

# Model training (metadata-driven)
assog2p train -j preprocessed/train_data_metadata.json -m LightGBM -o results/
```

## Detailed Usage ##
### 1. Data Preprocessing ###
**功能**：将VCF基因型数据与表型数据整合，生成机器学习模型输入文件

**输入文件要求**：
- **基因型文件**：标准VCF v4.2+格式，支持.gz压缩
- **表型文件**：制表符分隔的文本文件，包含两列（无表头）：
  ```
  sample_ID1 phenotype_value1
  sample_ID2 phenotype_value2
  ```

**完整参数列表**：
```bash
assog2p preprocess \
  -g <input.vcf> \
  -p <phenotype.txt> \
  -o <output_path> \
  [-f <feature_selection_mode>] \
  [--gwas_pvalue <threshold>] \
  [--ld-config "<window_kb>,<window_variants>,<r2_threshold>"] \
  [--no-filter-snps] \
  [--no-cache]
```

**参数说明**：
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `-g` | 必需 | - | 输入VCF文件路径 |
| `-p` | 必需 | - | 表型数据文件路径 |
| `-o` | 必需 | - | 输出路径（可为文件或目录） |
| `-f` | 可选 | 1 | 特征筛选模式（1=不筛选，2=GWAS，3=LD，4=GWAS+LD） |
| `--gwas_pvalue` | 可选 | 0.01 | GWAS P值阈值（`-f` 为 2 或 4 时生效） |
| `--ld-config` | 可选 | "50,5,0.2" | LD配置（`window_kb,window_variants,r2_threshold`，`-f` 为 3 或 4 时生效） |
| `--no-filter-snps` | 可选 | False | 关闭预处理阶段的SNP质量过滤 |
| `--no-cache` | 可选 | False | 禁用缓存 |

**输出文件格式**：
首行为表头，包含样本ID、所有SNP名称和表型列：
```
sample	rs12345	rs67890	...	phenotype
Sample1	0	1	...	0.56
Sample2	2	0	...	0.78
```

**表型文件示例**：

*二分类任务（binary classification）*：
```
1-1    0
1-2    0
1-3    1
2-1    1
```

*回归任务（regression）*：
```
1-1    9.5
1-2    7.4
1-3    10.8
2-1    12.5
```

**输出文件示例（output.txt）**：

| sample | snp1 | snp2 | ... | snp10000 | ... | snp... | phenotype |
|:-------|:-----|:-----|:-----|:---------|:-----|:-------|:----------|
| 1-1    | 1    | 0    | ... | 2        | ... | ...    | 0         |
| 1-2    | 0    | 1    | ... | 2        | ... | ...    | 0         |
| 1-3    | 1    | 1    | ... | 2        | ... | ...    | 1         |
| 2-1    | 0    | 2    | ... | 1        | ... | ...    | 1         |

**注意**：如果SNP值为缺失（None），则编码为 `-1`。


### 2. Model Training ###
**功能**：使用机器学习算法分析基因型与表型关联，计算特征重要性

**支持算法**：
| 算法名称 | 类型 | 特点 | 适用场景 |
|----------|------|------|----------|
| LightGBM | 分类/回归 | 高效快速，低内存 | 大规模数据集 |
| RandomForest | 分类/回归 | 鲁棒性强，不易过拟合 | 特征重要性评估 |
| XGBoost | 分类/回归 | 高精度，支持自定义损失函数 | 竞赛级模型 |
| SVM | 分类/回归 | 小样本表现好 | 高维特征空间 |
| CatBoost | 分类/回归 | 自动处理类别特征 | 含分类变量数据 |
| Logistic | 分类/回归 | 简单解释性强 | 基线模型对比 |

**说明**：训练阶段为 metadata 驱动，不再接收 GWAS/LD 筛选参数；特征筛选请在 `preprocess` 阶段通过 `-f/--gwas_pvalue/--ld-config` 完成。

**完整参数列表**：
```bash
assog2p train \
  -j <preprocess_metadata.json> \
  -m <algorithm> \
  -o <output_dir> \
  [--task_type <classification/regression>] \
  [--n_folds <number>] \
  [--random_state <seed>]
```

**参数说明**：
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `-j` | 必需 | - | 预处理元数据文件（`*_metadata.json`） |
| `-m` | 必需 | - | 算法名称（见上表） |
| `-o` | 必需 | - | 输出目录 |
| `--task_type` | 可选 | auto | 任务类型（自动检测或从元数据读取） |
| `--n_folds` | 可选 | 5 | 交叉验证折数 |
| `--random_state` | 可选 | 42 | 随机种子（重现结果） |

**输入文件格式**：
- `-j` 仅支持元数据文件（`*_metadata.json`），通常位于 preprocess 输出目录。

**输出文件结构**：
训练完成后，在输出目录下会生成以下文件：
```
output_dir/
├── {model_type}/              # 模型专属目录
│   ├── {model_type}_model.pkl # 训练好的模型文件
│   ├── {model_type}_metrics.json            # 评估指标
│   ├── selected_snps.txt      # 筛选的SNP列表
│   ├── {model_type}_feature_importance.txt # 特征重要性（完整列表）
│   ├── top_features.txt       # Top 100特征
│   ├── {model_type}_cv_results.json        # 交叉验证结果
│   ├── {model_type}_performance_curves.png  # 性能曲线图（分类任务）
│   ├── probability_distribution.png # 概率分布图（分类任务）
│   └── cv_training_curves.png # 交叉验证训练曲线
└── model_comparison_report.json # 全模型训练对比报告（train-all模式）
```

**注意**：
- GWAS和LD分析过程中生成的临时文件（如`output/`目录下的GWAS结果文件）会在模块完成后自动删除
- 所有重要的结果（如筛选的SNP列表、特征重要性等）都已保存在模型目录中，无需保留临时文件

**使用示例**：
```bash
# 单模型训练（空白对照，使用全部特征）
# 假设preprocess输出目录为preprocessed/，会生成train_data_metadata.json
assog2p train -j preprocessed/train_data_metadata.json -m LightGBM -o results/

# 全模型训练（训练所有支持的模型）
assog2p train-all -j preprocessed/train_data_metadata.json -o results/

# 推荐直接使用元数据文件输入
assog2p train -j preprocessed/train_data_metadata.json -m LightGBM -o results/
```


### 3. Result Visualization ###
**功能**：生成高质量全基因组特征重要性散点图，支持图片输出

**图表类型**：
- **散点图（Scatter Plot）**：全基因组特征重要性分布图，类似曼哈顿图风格
  - 显示所有特征在基因组上的位置和重要性值
  - 自动按染色体分组显示
  - 正负效应用不同颜色区分（蓝色=正效应，红色=负效应）

**完整参数列表**：
```bash
assog2p visualize \
  [-i <feature_importance.txt>] \
  [-I <plotting_data.npz>] \
  -o <output_prefix>
```

**参数说明**：
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `-i` | 可选 | - | 特征重要性数据文件（用于散点图） |
| `-I` | 可选 | - | 模型评估结果文件 `plotting_data.npz`（用于性能/CV可视化） |
| `-o` | 必需 | - | 输出文件前缀 |

**输入文件格式**：
支持三列格式的特征重要性文件：
- 列1：特征名（格式：染色体_位置，如`1_123456`）
- 列2：重要性绝对值（importance_abs或importance）
- 列3：正负效应（effect，值为1或-1）

也支持两列格式（特征名和重要性值，从数值推断正负）。

**交互式图表功能**：
- 鼠标悬停显示详细信息（染色体、位置、重要性值）
- 缩放和平移操作
- 自动计算并显示显著性阈值（99百分位数）

**使用示例**：
```bash
# 基于特征重要性文件生成散点图
assog2p visualize -i feature_importance.txt -o plot

# 基于 plotting_data.npz 生成性能/CV图
assog2p visualize -I LightGBM_plotting_data.npz -o plot
```



---

## Output Files ##

本节详细说明每个步骤输出的文件及其用途。

### 1. Preprocess Output ###

**命令**：`assog2p preprocess`

**输出文件列表**：

| 文件名 | 格式 | 说明 |
|--------|------|------|
| `{output_prefix}.txt` 或 `{output_prefix}.txt.gz` | 文本/压缩文本 | 预处理后的训练数据文件，包含样本ID、所有SNP特征和表型列 |
| `{output_prefix}_metadata.json` | JSON | 元数据文件，包含预处理信息（基因型格式、样本数、染色体列表、PLINK前缀等） |
| `phenotype_distribution_pie.png` | PNG | 表型分布饼图（仅分类任务，当类别数≤10时生成） |
| `phenotype_distribution_histogram.png` | PNG | 表型分布直方图（仅回归任务生成） |

**文件位置**：
- 主输出文件：用户指定的输出路径
- 元数据文件：与主输出文件同目录
- 图表文件：输出目录下（如果适用）

**示例**：
```bash
assog2p preprocess -g genotype.vcf -p phenotype.csv -o preprocessed_data
```
输出：
- `preprocessed_data.txt` - 训练数据
- `preprocessed_data_metadata.json` - 元数据
- `phenotype_distribution_*.png` - 表型分布图

---

### 2. Train Output ###

**命令**：`assog2p train`

**输出文件结构**：
```
{output_dir}/
└── {model_type}/                    # 模型专属目录（如 LightGBM/）
    ├── {model_type}_model.pkl      # 训练好的模型文件（用于预测）
    ├── {model_type}_metrics.json                 # 评估指标（准确率、AUC、R²等）
    ├── {model_type}_cv_results.json              # 交叉验证结果（每折的详细指标）
    ├── selected_snps.txt           # 筛选的SNP列表（GWAS/LD筛选后的特征）
    ├── {model_type}_feature_importance.txt       # 特征重要性完整列表（三列：feature, importance_abs, effect）
    ├── top_features.txt             # Top 100特征列表（仅特征名）
    ├── {model_type}_cv_training_curves.png      # 交叉验证训练过程曲线
    ├── performance_curves.png       # 性能评估曲线（ROC曲线/回归散点图）
    └── probability_distribution.png # 概率分布图（仅分类任务）
```

**文件说明**：

| 文件名 | 格式 | 内容说明 |
|--------|------|----------|
| `{model_type}_model.pkl` | Pickle | 训练好的模型对象，用于后续预测 |
| `{model_type}_metrics.json` | JSON | 包含平均准确率、AUC、R²、MAE等评估指标 |
| `{model_type}_cv_results.json` | JSON | 每折交叉验证的详细结果和平均指标 |
| `selected_snps.txt` | 文本 | 经过GWAS/LD筛选后的SNP列表（每行一个SNP名称） |
| `{model_type}_feature_importance.txt` | TSV | 三列格式：特征名、重要性绝对值、正负效应（1或-1） |
| `top_features.txt` | 文本 | Top 100重要特征名称列表（每行一个） |
| `{model_type}_cv_training_curves.png` | PNG | 交叉验证过程中训练集和验证集的损失/准确率变化曲线 |
| `performance_curves.png` | PNG | 分类任务：ROC曲线；回归任务：预测值vs真实值散点图 |
| `probability_distribution.png` | PNG | 分类任务：各类别的预测概率分布直方图 |

**临时文件说明**（当使用GWAS筛选时）：
- GWAS分析会在当前工作目录的`output/`目录下生成临时文件（包括关联分析结果、Kinship矩阵、协变量等）
- 这些临时文件会在模型训练完成后自动删除
- 显著SNP列表已保存在`selected_snps.txt`中，无需保留临时文件

**示例**：
```bash
assog2p train -j preprocessed_data_metadata.json -m LightGBM -o results/
```
输出目录结构：
```
results/
└── LightGBM/
    ├── LightGBM_model.pkl
    ├── LightGBM_metrics.json
    ├── LightGBM_cv_results.json
    ├── selected_snps.txt
    ├── LightGBM_feature_importance.txt
    ├── top_features.txt
    ├── LightGBM_cv_training_curves.png
    ├── performance_curves.png
    └── probability_distribution.png
```

---

### 3. Train-all Output ###

**命令**：`assog2p train-all`

**输出文件结构**：
```
{output_dir}/
├── LightGBM/                        # LightGBM模型输出（同train命令）
├── RandomForest/                    # RandomForest模型输出
├── XGBoost/                         # XGBoost模型输出
├── SVM/                             # SVM模型输出
├── CatBoost/                         # CatBoost模型输出
├── Logistic/                        # Logistic模型输出
└── model_comparison_report.json     # 所有模型的对比报告
```

**文件说明**：

| 文件名 | 格式 | 内容说明 |
|--------|------|----------|
| `model_comparison_report.json` | JSON | 包含所有模型的训练结果对比、任务类型、特征筛选模式、总训练时间等信息 |

每个模型目录下的文件与 `train` 命令相同（见上节）。

**示例**：
```bash
assog2p train-all -j preprocessed_data_metadata.json -o results/
```
输出：6个模型目录 + 1个对比报告文件

---

### 4. Predict Output ###

**命令**：`assog2p predict`

**输出文件列表**：

| 文件名 | 格式 | 说明 |
|--------|------|------|
| `{output_dir}/{model_type}/{model_type}_predictions.tsv` | TSV | 预测结果文件，包含样本ID和预测值 |

**文件格式**：

*分类任务*：
```
sample    prediction    prob_class_0    prob_class_1
Sample1   1             0.2             0.8
Sample2   0             0.9             0.1
```

*回归任务*：
```
sample    prediction
Sample1   9.5
Sample2   7.4
```

**示例**：
```bash
assog2p predict -i new_data.csv -m results/train/LightGBM/LightGBM_model.pkl -o results/
```
输出：
- `results/LightGBM/LightGBM_predictions.tsv` - 预测结果

---

### 5. Visualize Output ###

**命令**：`assog2p visualize`

**输出文件列表**：

| 文件名 | 格式 | 说明 |
|--------|------|------|
| `{output_prefix}_static.png` | PNG | 静态散点图（默认300 DPI，适合论文发表） |
| `{output_prefix}_interactive.html` | HTML | 交互式散点图（可缩放、悬停查看详情） |

**文件说明**：

- **静态图**：高分辨率PNG格式，显示全基因组特征重要性分布，包含染色体标注和正负效应颜色区分
- **交互式图**：HTML格式，支持鼠标悬停查看详细信息（染色体、位置、重要性值等）

**控制输出**：
- `-i`：输入特征重要性文件时生成散点图
- `-I`：输入 `plotting_data.npz` 时生成模型性能/CV图
- 同时提供 `-i` 与 `-I` 时，会在同一输出前缀下生成两类可视化结果

**示例**：
```bash
assog2p visualize -i feature_importance.txt -o plot
```
输出：
- `plot_static.png` - 静态散点图
- `plot_interactive.html` - 交互式散点图

---

## Sample Data ##
### 测试数据集
提供水稻4K品种的基因型和表型测试数据（约500MB）：

#### 获取数据
```bash
# 基因型数据（VCF格式，~380MB）
# 请从数据源下载VCF文件

# 表型数据（CSV格式，~2KB）
# 请从数据源下载表型文件

# 如果VCF文件是压缩格式，需要解压
gunzip genotype.vcf.gz
```

#### 文件说明
| 文件名 | 格式 | 内容描述 |
|--------|------|----------|
| rice4k_geno_add_del.vcf | VCF v4.2 | 4000个水稻品种的45万个SNP位点 |
| phenos.csv | CSV | 包含开花期、株高等6个表型性状 |

#### 数据使用示例
```bash
# 预处理示例（输出到目录，会生成train_data.txt）
assog2p preprocess -g rice4k_geno_add_del.vcf -p phenos.csv -o rice_preprocessed/

# 模型训练示例（空白对照模式）
assog2p train -j rice_preprocessed/train_data_metadata.json -m LightGBM -o rice_results/

# 可视化示例
assog2p visualize -i rice_results/LightGBM/feature_importance.txt -o manhattan
```

---

## FAQ ##
### 常见问题与解决方案

#### 1. 安装问题
**Q: 安装whl包时出现"Preparing metadata (pyproject.toml) ... error"或CMake版本错误**
这是依赖包（如xgboost）需要从源码编译，但CMake版本过低导致的。解决方法：

```bash
# 方法1：升级CMake（推荐）
conda activate bioenv
conda install cmake>=3.18 -c conda-forge
pip install assoG2P-1.0.0-py3-none-any.whl
```

```bash
# 方法2：使用预编译的依赖包
conda activate bioenv
pip install --prefer-binary xgboost lightgbm catboost
pip install --no-deps assoG2P-1.0.0-py3-none-any.whl
pip install numpy pandas scikit-learn matplotlib plotly kaleido seaborn shap numba
```

```bash
# 方法3：使用conda安装依赖（推荐）
conda activate bioenv
conda install -c conda-forge xgboost lightgbm catboost numpy pandas scikit-learn matplotlib plotly seaborn
pip install --no-deps assoG2P-1.0.0-py3-none-any.whl
pip install kaleido shap numba
```

```bash
# 方法4：清除pip缓存后重试（如果上述方法都失败）
# 有时pip缓存中的损坏文件会导致编译问题
conda activate bioenv
pip cache purge  # 清除所有pip缓存
pip install assoG2P-1.0.0-py3-none-any.whl
```

```bash
# 方法5：创建新的conda环境重新安装（推荐用于解决复杂编译问题）
# 确保使用符合要求的Python版本（3.8-3.12）
conda create -n assoG2P-new python=3.10  # 使用3.10作为示例，可根据需要选择3.8-3.12
conda activate assoG2P-new

# 安装CMake和构建工具
conda install cmake>=3.18 -c conda-forge
conda install -c conda-forge xgboost lightgbm catboost numpy pandas scikit-learn matplotlib plotly seaborn

# 安装whl包
pip install assoG2P-1.0.0-py3-none-any.whl

# 安装其他pip-only的依赖
pip install kaleido shap numba

# 验证安装
assog2p --version
```

**Q: 安装lightgbm时出现编译错误**
```bash
# 解决方案：安装预编译版本
pip install --prefer-binary lightgbm
```

**Q: 安装xgboost时出现CMake错误**
```bash
# 解决方案1：升级CMake
conda install cmake>=3.18 -c conda-forge

# 解决方案2：使用预编译版本
pip install --prefer-binary xgboost

# 解决方案3：清除pip缓存后重试
pip cache purge
pip install --prefer-binary xgboost

# 解决方案4：创建新环境重新安装（如果问题持续）
conda create -n assoG2P-new python=3.10
conda activate assoG2P-new
conda install cmake>=3.18 -c conda-forge
pip install --prefer-binary xgboost
```

#### 2. 运行时错误
**Q: 预处理大型VCF文件时内存不足**
```bash
# 解决方案：先仅做基础预处理，必要时拆分VCF后分批处理
assog2p preprocess -g large.vcf -p pheno.txt -o out.txt -f 1
```

**Q: 模型训练时报错"特征数量超过限制"**
```bash
# 解决方案：在 preprocess 阶段使用 GWAS/LD 进行筛选，再用 metadata 训练
assog2p preprocess -g data.vcf -p pheno.txt -o preprocessed/ -f 2 --gwas_pvalue 0.01
assog2p train -j preprocessed/train_data_metadata.json -m LightGBM -o results/
```

**Q: 训练阶段如何使用特征筛选？**
```bash
# 先在 preprocess 阶段设置筛选模式
assog2p preprocess -g data.vcf -p pheno.txt -o preprocessed/ -f 4 --gwas_pvalue 0.01 --ld-config "50,5,0.2"
# 再使用 metadata 进行训练
assog2p train -j preprocessed/train_data_metadata.json -m LightGBM -o results/
```

**Q: GWAS和LD综合筛选的流程是什么？**
- 模式4（GWAS和LD综合）：先执行GWAS分析，筛选出显著SNP（P值阈值以下），然后仅对这些GWAS显著SNP进行LD过滤，最终直接使用LD过滤结果作为特征进行模型训练。若LD过滤失败或无结果，则退回到仅使用GWAS显著SNP。

**Q: GWAS分析生成的output/目录下的文件会被保留吗？**
- 不会。所有GWAS分析过程中生成的临时文件（包括`output/`目录下的关联分析结果、Kinship矩阵、协变量等）会在模型训练完成后自动删除。
- 所有重要的结果（如筛选的SNP列表、特征重要性等）都已保存在模型目录中，无需保留临时文件。

#### 3. 数据格式问题
**Q: VCF文件解析错误"invalid chromosome format"**
```bash
# 解决方案：检查染色体命名格式是否为数字或字母+数字
# 正确示例：chr1、1、chrUn；错误示例：Chr01、chrI
```

#### 4. 可视化问题
**Q: 可视化时应该使用哪些输入参数？**
```bash
# 特征重要性散点图
assog2p visualize -i feature_importance.txt -o plot
# 模型性能/CV图
assog2p visualize -I plotting_data.npz -o plot
```

---

## Developer Guide ##
### Contribution Guidelines
欢迎通过以下方式贡献代码：
1. Fork本仓库并创建分支（`git checkout -b feature/amazing-feature`）
2. 提交修改（`git commit -m 'Add some amazing feature'`）
3. 推送到分支（`git push origin feature/amazing-feature`）
4. 创建Pull Request

### Development Environment Setup
```bash
# 克隆仓库
git clone https://github.com/chenrf/assoG2P.git
cd assoG2P

# 创建开发环境
conda create -n assoG2P-dev python=3.9
conda activate assoG2P-dev
pip install -e .[dev]

# 运行测试
pytest tests/
```

### Code Style Requirements
- 遵循PEP 8规范
- 使用类型注解
- 添加单元测试（覆盖率>80%）
- 提交前运行`black`格式化代码

## License ##
本项目采用MIT许可证：
```
MIT License

Copyright (c) 2023 assoG2P Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Code Structure**
```text
assoG2P/  
├── bin/                    # Core modules  
│   ├── preprocess.py       # Data preprocessing  
│   ├── modeltraining.py    # Model training (with GWAS/LD integration)
│   ├── gemma_gwas.py       # GWAS analysis module
│   ├── plink_ld.py         # LD filtering module
│   └── visualization.py    # Visualization  
├── main.py                 # Main entry  
└── setup.py                # Package configuration  
```