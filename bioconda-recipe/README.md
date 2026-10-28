# Bioconda 发布（G2PInsight）

本目录只服务于 **官方 bioconda**：把 `meta.yaml` 复制到 [bioconda-recipes](https://github.com/bioconda/bioconda-recipes) 的 `recipes/g2pinsight/` 后提 PR。

个人频道请用仓库里的 `conda-recipe/`，不要混用。

当前版本：**1.1.2**（源为 PyPI sdist，sha256 已写入 `meta.yaml`）。

run 依赖版本与 `pyproject.toml` 中 **Python &lt; 3.13** 对齐（含 `xgboost >=1.6,<2.1` 等）。Python 3.13 请用 pip。

---

## 第 0 步：前置检查

- [ ] 已发布到 PyPI：`pip install G2PInsight==1.1.2`
- [x] sdist 含 `LICENSE`
- [x] 入口脚本：`G2PInsight`
- [x] 本目录 `meta.yaml` 使用 `url` + `sha256`（非本地 `path`）

可选（非首次必需）：之后可再 bump，让运行时优先用 PATH / bioconda 的 `plink`、`gemma`，再回退包内二进制。

---

## 第 1 步：Fork bioconda-recipes

1. 打开 https://github.com/bioconda/bioconda-recipes  
2. **Fork**（建议账号 `ZBaoyi`，与 `recipe-maintainers` 一致）  
3. 得到：`https://github.com/ZBaoyi/bioconda-recipes`

---

## 第 2 步：Clone 并建分支

```bash
git clone --depth 1 https://github.com/ZBaoyi/bioconda-recipes.git
cd bioconda-recipes
git remote add upstream https://github.com/bioconda/bioconda-recipes.git
git fetch upstream
git checkout -b add-g2pinsight upstream/master
```

---

## 第 3 步：放入 recipe

```bash
mkdir -p recipes/g2pinsight
cp /share/home/students/chenrf/assocG2P/bioconda-recipe/meta.yaml recipes/g2pinsight/meta.yaml
```

确认只有：

```text
recipes/g2pinsight/meta.yaml
```

---

## 第 4 步（可选）：本地 lint / 构建

需要 Docker；也可跳过，直接靠 CI。

```bash
conda create -n bioconda -c conda-forge -c bioconda bioconda-utils -y
conda activate bioconda

bioconda-utils lint --packages g2pinsight
bioconda-utils build --docker --mulled-build-and-test --packages g2pinsight
```

| 现象 | 处理 |
|------|------|
| 依赖名不存在 | 改成 conda-forge/bioconda 上的名字（如 `python-kaleido`） |
| sha256 不对 | 用下面命令重算并更新 |
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

1. fork 页点 **Compare & pull request**  
2. base：`bioconda/bioconda-recipes` → `master`  
3. compare：`add-g2pinsight`  
4. 标题：`Add g2pinsight`  
5. 正文：用途、PyPI 链接、你是 upstream maintainer  

---

## 第 7 步：CI 与审核

1. 等 CI 变绿  
2. PR 评论：

```text
@BiocondaBot please add label
```

3. merge 后自动上 bioconda  

用户安装：

```bash
conda install -c bioconda -c conda-forge g2pinsight
G2PInsight --version
```

---

## 以后发新版本

1. 先发新版到 PyPI  
2. 更新本目录（或 bioconda-recipes 里）`meta.yaml`：`version`、`sha256`、`build.number=0`  
3. 开 PR：`Update g2pinsight to x.y.z`

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
