# assoG2P Genomic Analysis Tool

## Table of Contents
- [Feature Overview](#feature-overview)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Result Visualization](#3-result-visualization)
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
  - 超参数自动优化
  - 特征重要性计算与排序

### 3. Result Visualization
- **功能**：生成 publication 级别的基因组数据可视化图表
- **图表类型**：曼哈顿图、QQ图、特征重要性条形图、染色体分布散点图
- **核心特性**：
  - 静态（PNG/SVG）和交互式（HTML）输出
  - 自动计算显著性阈值
  - 染色体位置自动标注
  - 支持批量图表生成与比较
---

## Installation Guide ##
### System Requirements
- Python 3.8-3.11 (64-bit)
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

进入该环境:
```conda activate bioenv```

然后进行基因型数据的筛选、过滤、质量控制：
```bash
plink 

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
association --version
# 应显示 assoG2P 1.0.0
```

### 常见安装问题
- 权限问题：Linux/macOS添加`sudo`前缀
```pip install assog2p-1.0.0-py3-none-any.whl```
The following message indicates successful installation：

```bash
Installing collected packages : assog2p
Successfully installed assog2p-1.0.0
```

After installation, ```association``` is registered as a command-line tool in the current environment.
Run ```association -h ```to view usage help.

---

## Quick Start ##
**Complete Workflow Example**

```bash
# Data preprocessing
association preprocess -g genotype.vcf -p phenotype.txt -o ml.txt

#  Model training (using LightGBM as an example)  
association train -i ml.txt -m LightGBM -o importance.csv

# Feature importance visualization  
association visualize -i importance.csv -o result.png -t scatter
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
association preprocess \
  -g <input.vcf> \
  -p <phenotype.txt> \
  -o <output.txt> \
  [--filter-missing <threshold>] \
  [--impute-method <mean/median>] \
  [--max-samples <number>] \
  [--threads <number>]
```

**参数说明**：
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `-g` | 必需 | - | 输入VCF文件路径 |
| `-p` | 必需 | - | 表型数据文件路径 |
| `-o` | 必需 | - | 输出文件路径 |
| `--filter-missing` | 可选 | 0.1 | 样本缺失率阈值（>此值将被过滤） |
| `--impute-method` | 可选 | mean | 缺失值填充方法（mean/median） |
| `--max-samples` | 可选 | 500000 | 最大样本量（超过将随机抽样） |
| `--threads` | 可选 | 4 | 并行处理线程数 |

**输出文件格式**：
首行为表头，包含样本ID、所有SNP名称和表型列：
```
sample	rs12345	rs67890	...	phenotype
Sample1	0	1	...	0.56
Sample2	2	0	...	0.78
```

**phenotype file example if it's a binary task** :


|1-1|0|
|:---|:---|
|1-2|0|
|1-3|1|
|2-1|1|

phenotype file example **if it's a regression task**:


|1-1|9.5|
|:---|:---|
|1-2|7.4|
|1-3|10.8|
|2-1|12.5|

output.txt example:

|sample|snp1|snp2|...|snp10000|...|snp...|phenotype|
|:---|:---|:---|:---|:---|:---|:---|:---|
|1-1|1|0|...|2|...|...|0|
|1-2|0|1|...|2|...|...|0|
|1-3|1|1|...|2|...|...|1|
|2-1|0|2|...|1|...|...|1|

**_If a SNP is None, it is encoded as -1**_


### 2. Model Training ###
**功能**：使用机器学习算法分析基因型与表型关联，计算特征重要性

**支持算法**：
| 算法名称 | 类型 | 特点 | 适用场景 |
|----------|------|------|----------|
| LightGBM | 分类/回归 | 高效快速，低内存 | 大规模数据集 |
| RandomForest | 分类/回归 | 鲁棒性强，不易过拟合 | 特征重要性评估 |
| XGBoost | 分类/回归 | 高精度，支持自定义损失函数 | 竞赛级模型 |
| SVM | 分类 | 小样本表现好 | 高维特征空间 |
| CatBoost | 分类/回归 | 自动处理类别特征 | 含分类变量数据 |
| Logistic | 分类 | 简单解释性强 | 基线模型对比 |

**完整参数列表**：
```bash
association train \
  -i <input.txt> \
  -m <algorithm> \
  -o <output.csv> \
  [--task-type <classification/regression>] \
  [--cv-folds <number>] \
  [--hyperopt <iterations>] \
  [--test-size <ratio>] \
  [--random-state <seed>]
```

**参数说明**：
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `-i` | 必需 | - | 预处理后的输入文件 |
| `-m` | 必需 | - | 算法名称（见上表） |
| `-o` | 必需 | - | 特征重要性输出文件 |
| `--task-type` | 可选 | auto | 任务类型（自动检测） |
| `--cv-folds` | 可选 | 5 | 交叉验证折数 |
| `--hyperopt` | 可选 | 20 | 超参数优化迭代次数 |
| `--test-size` | 可选 | 0.2 | 测试集比例 |
| `--random-state` | 可选 | 42 | 随机种子（重现结果） |

-i input.txt **The format is the same as the output file format of the previous step**

-o importance.csv example:

|Feature|Importance|
|:---|:---|
|snp1|0.45|
|snp2|0.21|
|snp...|...|
|snp...|...|


### 3. Result Visualization ###
**功能**：生成高质量基因组数据可视化图表，支持 publication 级图片输出

**图表类型与应用场景**：
| 参数类型 | 图表名称 | 适用场景 | 输出格式 |
|----------|----------|----------|----------|
| scatter | 曼哈顿图 | 全基因组关联分析结果 | PNG/SVG/HTML |
| bar | 染色体汇总图 | 染色体水平特征比较 | PNG/SVG |
| line | 趋势图 | 基因组区域信号变化 | PNG/SVG |
| hist | 分布图 | 特征值分布统计 | PNG/SVG |
| manhattan | 增强曼哈顿图 | 显著SNP标记显示 | HTML (交互式) |

**完整参数列表**：
```bash
association visualize \
  -i <input.csv> \
  -o <output.[png/svg/html]> \
  -t <chart_type> \
  [--title <plot_title>] \
  [--threshold <p_value>] \
  [--interactive] \
  [--dpi <resolution>] \
  [--color-map <palette>] \
  [--chromosome <number>]
```

**参数说明**：
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `-i` | 必需 | - | 输入文件（模型输出的CSV） |
| `-o` | 必需 | - | 输出文件路径（格式自动识别） |
| `-t` | 必需 | - | 图表类型（scatter/bar/line/hist/manhattan） |
| `--title` | 可选 | "Association Results" | 图表标题 |
| `--threshold` | 可选 | 0.05 | 显著性阈值（点标记为红色） |
| `--interactive` | 可选 | False | 启用交互式HTML图表 |
| `--dpi` | 可选 | 300 | 图像分辨率（仅静态图） |
| `--color-map` | 可选 | "viridis" | 颜色方案（matplotlib兼容） |
| `--chromosome` | 可选 | 全部 | 指定染色体（如1/3/X） |

**交互式图表功能**：
- 悬停显示详细SNP信息（ID、位置、P值）
- 缩放和平移操作
- 点击下载高分辨率图像
- 特征筛选和高亮功能

**使用示例**：
```bash
# 基础曼哈顿图
association visualize -i importance.csv -o manhattan.png -t scatter --dpi 300

# 交互式全基因组视图
association visualize -i results.csv -o interactive.html -t manhattan --interactive --threshold 0.01

# 染色体1特定区域可视化
association visualize -i region.csv -o chr1.png -t line --chromosome 1 --title "Chromosome 1 Analysis"
```



---

## Sample Data ##
### 测试数据集
提供水稻4K品种的基因型和表型测试数据（约500MB）：

#### 获取数据
```bash
# 基因型数据（VCF格式，~380MB）
wget 

# 表型数据（CSV格式，~2KB）
wget 

# 解压VCF文件
gunzip 
```

#### 文件说明
| 文件名 | 格式 | 内容描述 |
|--------|------|----------|
| rice4k_geno_add_del.vcf | VCF v4.2 | 4000个水稻品种的45万个SNP位点 |
| phenos.csv | CSV | 包含开花期、株高等6个表型性状 |

#### 数据使用示例
```bash
# 预处理示例
bashassociation preprocess -g rice4k_geno_add_del.vcf -p phenos.csv -o rice_ml.txt --filter-missing 0.05

# 模型训练示例
association train -i rice_ml.txt -m LightGBM -o rice_importance.csv --task-type regression

# 可视化示例
association visualize -i rice_importance.csv -o manhattan.html -t scatter --interactive
```

---

## FAQ ##
### 常见问题与解决方案

#### 1. 安装问题
**Q: 安装lightgbm时出现编译错误**
```bash
# 解决方案：安装预编译版本
pip install --prefer-binary lightgbm
```

#### 2. 运行时错误
**Q: 预处理大型VCF文件时内存不足**
```bash
# 解决方案：启用抽样功能
association preprocess -g large.vcf -p pheno.txt -o out.txt --max-samples 300000
```

**Q: 模型训练时报错"特征数量超过限制"**
```bash
# 解决方案：增加内存限制或减少特征数量
association train -i data.txt -m LightGBM -o imp.csv --hyperopt 10
```

#### 3. 数据格式问题
**Q: VCF文件解析错误"invalid chromosome format"**
```bash
# 解决方案：检查染色体命名格式是否为数字或字母+数字
# 正确示例：chr1、1、chrUn；错误示例：Chr01、chrI
```

#### 4. 可视化问题
**Q: 生成的散点图点重叠严重**
```bash
# 解决方案：启用点大小自适应或减少样本量
association visualize -i data.csv -o plot.png -t scatter --adjust-size
```pip install assog2p-1.0.0-py3-none-any.whl  
```

**How to Generate a PDF Report?**

```bash
association visualize -i results.csv -o report.pdf -t bar --dpi 300
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
├── bin/                # Core modules  
│   ├── preprocess.py   # Data preprocessing  
│   ├── modeltraining.py # Model training  
│   └── visualization.py # Visualization  
├── main.py             # Main entry  
└── setup.py            # Package configuration  
```