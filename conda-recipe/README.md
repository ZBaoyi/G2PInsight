# 个人 conda 频道（anaconda.org/zbaoyi）

本目录只服务于 **个人频道** `zbaoyi`：本地 `conda build` 后 `anaconda upload`。

官方 bioconda 请用 `bioconda-recipe/`，不要混用。

当前版本：**1.1.1**

依赖均来自 **conda-forge**（不需要 `-c bioconda`）。PLINK/GEMMA 使用包内自带二进制。

---

## 前置

```bash
# 需要已发布到 PyPI 的同版本 sdist（meta.yaml 用 url + sha256）
conda install -c conda-forge conda-build anaconda-client -y
anaconda login   # 账号：zbaoyi
```

本机若是 CentOS 7 / glibc 过旧，系统自带的 `conda build` 常会失败；用下面「独立 cbuild 环境」构建。

---

## 构建（推荐：独立 micromamba 环境）

一次性准备（装在 `/data_p`，避免占满系统盘）：

```bash
# micromamba 二进制若已有可跳过
# curl 或从 GitHub 下载 micromamba-linux-64 → ~/bin/micromamba

export MAMBA_ROOT_PREFIX=/data_p/students/chenrf/micromamba-root
micromamba create -y -n cbuild -c conda-forge \
  conda-build anaconda-client conda-libmamba-solver python=3.12
```

每次构建：

```bash
cd /share/home/students/chenrf/assocG2P
mkdir -p /data_p/students/chenrf/conda-bld-out

export CONDA_OVERRIDE_GLIBC=2.34   # CentOS 7 需要，否则依赖求解失败
export CONDA_OVERRIDE_LINUX=4.18
export PATH=/data_p/students/chenrf/micromamba-root/envs/cbuild/bin:/usr/bin:/bin
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE

conda build conda-recipe -c conda-forge --override-channels \
  --output-folder /data_p/students/chenrf/conda-bld-out \
  --no-anaconda-upload --no-test
```

产物：`/data_p/students/chenrf/conda-bld-out/noarch/g2pinsight-1.1.1-py_0.conda`

（较新系统也可直接 `conda build conda-recipe -c conda-forge --override-channels`，一般不必设 `CONDA_OVERRIDE_*`。）

---

## 上传

```bash
anaconda upload -u zbaoyi \
  /data_p/students/chenrf/conda-bld-out/noarch/g2pinsight-1.1.1-py_0.conda
```

---

## 用户安装

```bash
conda install -c zbaoyi -c conda-forge g2pinsight
G2PInsight --version
```

---

## 发新版本时

1. 改版本号并发布到 PyPI  
2. 更新本目录 `meta.yaml`：`version`、`sha256`、`build.number=0`  
3. 再构建 + `anaconda upload`
