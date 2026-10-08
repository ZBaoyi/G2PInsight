# Bioconda 发布步骤（G2PInsight）

本目录的 `meta.yaml` 可直接复制到 [bioconda-recipes](https://github.com/bioconda/bioconda-recipes) 的 `recipes/g2pinsight/`。

当前对应 PyPI 版本：**1.0.8**（sha256 已写入 `meta.yaml`）。

---

## 第 0 步：前置检查（本仓库已就绪）

- [x] 已发布到 PyPI：`pip install G2PInsight==1.0.8`
- [x] sdist 含 `LICENSE`
- [x] 入口脚本：`G2PInsight`
- [x] 本目录已有 `meta.yaml`

可选（更好体验，非首次提交必需）：让代码 **优先用 PATH 里的 plink/gemma**，再回退包内二进制；发新版到 PyPI 后再 bump bioconda。

---

## 第 1 步：Fork bioconda-recipes

1. 浏览器打开：https://github.com/bioconda/bioconda-recipes  
2. 点右上角 **Fork**（账号建议用 `ZBaoyi`，与 `recipe-maintainers` 一致）  
3. Fork 完成后，得到：`https://github.com/ZBaoyi/bioconda-recipes`

---

## 第 2 步：Clone 并建分支

```bash
git clone --depth 1 https://github.com/ZBaoyi/bioconda-recipes.git
cd bioconda-recipes
git remote add upstream https://github.com/bioconda/bioconda-recipes.git
git fetch upstream
git checkout -b add-g2pinsight upstream/master
```

若你 clone 的是自己的 fork 且默认分支已是最新：

```bash
git checkout -b add-g2pinsight
```

---

## 第 3 步：放入 recipe

```bash
mkdir -p recipes/g2pinsight
# 把本仓库的 meta.yaml 拷过去（按你的实际路径改）
cp /path/to/assocG2P/bioconda-recipe/meta.yaml recipes/g2pinsight/meta.yaml
```

确认文件为：

```text
recipes/g2pinsight/meta.yaml
```

若要改 maintainer，编辑 `extra.recipe-maintainers` 里的 GitHub 用户名。

---

## 第 4 步（可选）：本地 lint / 构建

需要 Docker，且环境稍重；也可跳过，直接靠 GitHub CI。

```bash
conda create -n bioconda -c conda-forge -c bioconda bioconda-utils -y
conda activate bioconda

# 在 bioconda-recipes 根目录
bioconda-utils lint --packages g2pinsight

# 有 Docker 时再构建测试
bioconda-utils build --docker --mulled-build-and-test --packages g2pinsight
```

常见失败：

| 现象 | 处理 |
|------|------|
| 依赖名不存在 | 改成 conda-forge/bioconda 上的名字（如 `python-kaleido`） |
| sha256 不对 | 重新算 sdist 哈希并更新 |
| lint 要求 `run_exports` | 本 `meta.yaml` 已包含 |

---

## 第 5 步：提交并推送

```bash
git add recipes/g2pinsight/meta.yaml
git commit -m "Add g2pinsight"
git push -u origin add-g2pinsight
```

---

## 第 6 步：开 Pull Request

1. 打开 fork 页，GitHub 通常会提示 **Compare & pull request**  
2. base：`bioconda/bioconda-recipes` → `master`  
3. compare：你的 `add-g2pinsight`  
4. 标题：`Add g2pinsight`  
5. 正文可简短写：包用途、PyPI 链接、你是 upstream maintainer  

---

## 第 7 步：等 CI，然后请求审核

1. 等 Azure / GitHub CI 变绿（红了就按日志改 `meta.yaml` 再 push）  
2. 在 PR 评论里发：

```text
@BiocondaBot please add label
```

3. 审核通过并 merge 后，包会自动上传到 bioconda  

用户安装：

```bash
conda install -c bioconda -c conda-forge g2pinsight
G2PInsight --version
```

---

## 以后发新版本

1. 先发新版到 PyPI（改 `pyproject.toml` / `__version__`）  
2. 在 bioconda-recipes 里改 `recipes/g2pinsight/meta.yaml`：  
   - `version`  
   - `sha256`（新 sdist）  
   - `build.number` 重置为 `0`  
3. 再开 PR：`Update g2pinsight to x.y.z`

算 sha256：

```bash
curl -sL -o /tmp/g2pinsight.tar.gz \
  https://pypi.io/packages/source/g/g2pinsight/g2pinsight-VERSION.tar.gz
sha256sum /tmp/g2pinsight.tar.gz
```

---

## 官方文档

- [Contribution workflow](https://bioconda.github.io/contributor/workflow.html)  
- [Adding bioinformatic software](https://bioconda.github.io/tutorials/2024-adding-bioinformatic-software-to-bioconda.html)  
- [Guidelines](https://bioconda.github.io/contributor/guidelines.html)  
