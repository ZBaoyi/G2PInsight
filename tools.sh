#!/bin/bash

# 临时文件管理机制
TEMP_FILES=()
cleanup_temp_files() {
    if [ ${#TEMP_FILES[@]} -gt 0 ]; then
        echo -e "${YELLOW}>>> 清理临时文件...${NC}"
        rm -f "${TEMP_FILES[@]}" >/dev/null 2>&1
    fi
}
trap cleanup_temp_files EXIT

set -eo pipefail

# =============== 用户配置区 ===============
TOOLS=(
    "plink"         # 全基因组关联分析
    "fastqc"        # 质控检查
    "bwa"           # 序列比对
    "samtools"      # SAM工具集
    "bcftools"      # VCF处理
    "bedtools"      # 基因组算术
    "blast"         # 序列比对
    "bowtie2"       # 短序列比对
)

CONDA_PATH="${HOME}/bioconda"   # 安装目录
ENV_NAME="bioenv"               # 环境名称

# =============== 核心函数 ===============
init_colors() {
    RED='\033[1;31m'; GREEN='\033[1;32m'
    YELLOW='\033[1;33m'; BLUE='\033[1;34m'
    NC='\033[0m'
}

show_header() {
    echo -e "${BLUE}"
    echo -e "=============================================="
    echo -e "          生物信息学工具安装脚本              "
    echo -e "==============================================${NC}"
    echo -e "安装路径: ${YELLOW}${CONDA_PATH}${NC}"
    echo -e "环境名称: ${YELLOW}${ENV_NAME}${NC}"
    echo -e "工具数量: ${YELLOW}${#TOOLS[@]}${NC}个"
    echo -e "==============================================\n${NC}"
}


check_deps() {
    local missing=()
    for cmd in wget curl unzip; do
        if ! command -v $cmd &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}缺少依赖: ${missing[*]}${NC}"
        echo -e "${YELLOW}尝试自动安装依赖...(需要sudo权限)${NC}"
        
        if command -v apt-get &>/dev/null; then
            sudo apt-get update && sudo apt-get install -y ${missing[@]} || {
                echo -e "${RED}依赖安装失败，请手动安装后重试${NC}"
                exit 1
            }
        elif command -v yum &>/dev/null; then
            sudo yum install -y ${missing[@]} || {
                echo -e "${RED}依赖安装失败，请手动安装后重试${NC}"
                exit 1
            }
        else
            echo -e "${RED}无法自动安装依赖，请手动安装: ${missing[*]}${NC}"
            exit 1
        fi
    fi
}

install_miniconda() {
    local install_url="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    local temp_file="/tmp/miniconda.sh"
    TEMP_FILES+=($temp_file)

    echo -e "${YELLOW}>>> 正在安装Miniconda到 ${CONDA_PATH}${NC}"
    mkdir -p "$(dirname "${CONDA_PATH}")"
    
    if command -v wget &>/dev/null; then
        wget --show-progress -qO "$temp_file" "$install_url"
    else
        curl -# -L "$install_url" -o "$temp_file"
    fi

    bash /tmp/miniconda.sh -b -p "${CONDA_PATH}" && rm /tmp/miniconda.sh || {
        echo -e "${RED}Miniconda安装失败!${NC}"
        exit 1
    }

    # 初始化conda
    export PATH="${CONDA_PATH}/bin:$PATH"
    eval "$("${CONDA_PATH}/bin/conda" shell.bash hook)"
    conda init &>/dev/null
}

setup_channels() {
    echo -e "${YELLOW}>>> 配置清华镜像源${NC}"
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda
    conda config --set show_channel_urls yes
    conda clean -i -y
}

create_env() {
    if conda env list | grep -q "${ENV_NAME}"; then
        echo -e "${YELLOW}>>> 环境 ${ENV_NAME} 已存在，跳过创建${NC}"
        return
    fi

    echo -e "${YELLOW}>>> 创建隔离环境: ${ENV_NAME}${NC}"
    conda create -y -n "${ENV_NAME}" || {
        echo -e "${RED}环境创建失败!${NC}"
        exit 1
    }
}

install_tools() {
    local PROGRESS_FILE=".install_progress"
    
    # 进度保存机制 - 恢复安装状态
    if [ -f "$PROGRESS_FILE" ]; then
        echo -e "${YELLOW}>>> 检测到安装进度文件，恢复之前的安装状态${NC}"
        source "$PROGRESS_FILE"
        echo -e "${GREEN}已完成: ${#installed[@]}个工具，跳过已安装项${NC}"
    else
        local installed=()
        local failed=()
    fi
    
    echo -e "${YELLOW}>>> 安装生物信息工具 (共${#TOOLS[@]}个)${NC}"
    
    # 激活环境
    source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
    
    for tool in "${TOOLS[@]}"; do
        # 跳过已安装工具
        if [[ "${installed[@]}" =~ "${tool}" ]]; then
            echo -e "${BLUE}▸ ${tool} 已安装，跳过...${NC}"
            continue
        fi
        
        echo -e "${BLUE}▸ 正在安装 ${tool}...${NC}"
        if conda install -y -n "${ENV_NAME}" "${tool}"; then
            installed+=("$tool")
            echo -e "${GREEN}  ✓ ${tool} 安装成功${NC}"
            
            # 保存安装进度
            echo "installed=(${installed[@]})" > "$PROGRESS_FILE"
            echo "failed=(${failed[@]})" >> "$PROGRESS_FILE"
        else
            failed+=("$tool")
            echo -e "${RED}  ✗ ${tool} 安装失败${NC}"
        fi
    done
    
    # 显示汇总报告
    echo -e "\n${GREEN}======== 安装报告 ========${NC}"
    echo -e "成功: ${GREEN}${#installed[@]}${NC} 个"
    echo -e "失败: ${RED}${#failed[@]}${NC} 个"
    
    if [ ${#failed[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}以下工具安装失败:${NC}"
        printf ' - %s\n' "${failed[@]}"
        echo -e "\n${YELLOW}您可以尝试手动安装:"
        echo -e "conda install -n ${ENV_NAME} 工具名${NC}"
    fi
    
    conda deactivate

    # 清理进度文件
    if [ -f "$PROGRESS_FILE" ]; then
        rm -f "$PROGRESS_FILE"
    fi
}

show_usage() {
    echo -e "${GREEN}
使用方法:
  source ${CONDA_PATH}/bin/activate ${ENV_NAME}

常用命令:
  conda list              查看已安装工具
  conda update --all      更新所有工具
  conda remove 工具名     卸载特定工具

环境管理:
  conda env export > environment.yml  导出环境配置
  conda env create -f environment.yml 从配置恢复环境${NC}"
}

# =============== 主流程 ===============
main() {
    init_colors
    show_header
    check_deps
    
    if conda_path=$(command -v conda); then
        CONDA_PATH=$(dirname "$(dirname "$conda_path")")
        echo -e "${GREEN}>>> 检测到已安装Conda: ${CONDA_PATH}${NC}"
    else
        install_miniconda
    fi
    
    export PATH="${CONDA_PATH}/bin:$PATH"
    eval "$("${CONDA_PATH}/bin/conda" shell.bash hook)"
    
    setup_channels
    create_env
    install_tools
    
    echo -e "\n${GREEN}=== 安装完成! ===${NC}"
    show_usage
}

main "$@"