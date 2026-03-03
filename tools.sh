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
    "plink2"
    "cmake>=3.18"   # 构建工具（xgboost等包的编译依赖，需要3.18或更高版本）
)

CONDA_PATH="${HOME}/bioconda"   # 安装目录
ENV_NAME="bioenv"               # 环境名称
INSTALL_PYTHON_DEPS=false       # 是否安装 Python 依赖（assoG2P 包）
WHL_FILE=""                     # whl 文件路径（如果提供，则安装 Python 包）

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
        echo -e "${YELLOW}请手动安装以下依赖后重试: ${missing[*]}${NC}"
        exit 1
    fi
}

install_miniconda() {
    local install_url="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    local temp_file="/tmp/miniconda.sh"
    TEMP_FILES+=($temp_file)

    echo -e "${YELLOW}>>> 正在安装Miniconda到 ${CONDA_PATH}${NC}"
    mkdir -p "$(dirname "${CONDA_PATH}")"
    
    if command -v curl &>/dev/null; then
        curl -# -L "$install_url" -o "$temp_file"
    elif command -v wget &>/dev/null; then
        wget -qO "$temp_file" "$install_url"
    else
        echo -e "${RED}错误: 系统中未找到wget或curl工具${NC}"
        exit 1
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

setup_conda_channels() {
    echo -e "${YELLOW}>>> 检查并配置清华镜像源${NC}"
    
    # 检查是否已配置清华镜像源
    local tsinghua_main="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
    local tsinghua_free="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free"
    local tsinghua_conda_forge="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
    local tsinghua_bioconda="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda"
    
    local channels_configured=0
    local channel_list=$(conda config --show channels 2>/dev/null || echo "")
    
    # 检查是否已包含清华镜像源
    if echo "$channel_list" | grep -q "mirrors.tuna.tsinghua.edu.cn"; then
        echo -e "${GREEN}>>> 检测到已配置清华镜像源，跳过配置${NC}"
        channels_configured=1
    fi
    
    # 如果未配置，则添加镜像源
    if [ "$channels_configured" -eq 0 ]; then
        echo -e "${YELLOW}>>> 正在配置清华镜像源...${NC}"
        conda config --add channels "$tsinghua_main" 2>/dev/null || true
        conda config --add channels "$tsinghua_free" 2>/dev/null || true
        conda config --add channels "$tsinghua_conda_forge" 2>/dev/null || true
        conda config --add channels "$tsinghua_bioconda" 2>/dev/null || true
        conda config --set show_channel_urls yes 2>/dev/null || true
        echo -e "${GREEN}>>> 清华镜像源配置完成${NC}"
    fi
    
    # 清理索引缓存（无论是否新配置都执行，确保使用最新镜像）
    conda clean -i -y 2>/dev/null || true
}

detect_current_env() {
    # 检测当前是否在 conda 环境中
    # 如果 CONDA_DEFAULT_ENV 存在且不是 base，说明在某个环境中
    if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
        echo -e "${GREEN}>>> 检测到当前已在 conda 环境中: ${CONDA_DEFAULT_ENV}${NC}"
        ENV_NAME="$CONDA_DEFAULT_ENV"
        return 0  # 在环境中
    else
        return 1  # 不在环境中
    fi
}

prompt_for_env_name() {
    read -p "${YELLOW}请输入新的环境名称: ${NC}" new_name
    if [ -z "$new_name" ]; then
        echo -e "${RED}环境名称不能为空!${NC}"
        prompt_for_env_name
    elif conda env list | grep -q "$new_name"; then
        echo -e "${RED}环境 $new_name 已存在!${NC}"
        prompt_for_env_name
    else
        ENV_NAME="$new_name"
    fi
}

check_tool_executable() {
    # 检查工具是否可执行
    # 参数: $1 = 工具名, $2 = "no_activate" 表示环境已激活
    local tool_name="$1"
    local need_deactivate=0
    
    if [[ "$2" != "no_activate" ]]; then
        source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
        need_deactivate=1
    fi
    
    # 方法1: 使用 conda run（推荐）
    if conda run -n "${ENV_NAME}" which "$tool_name" &>/dev/null; then
        if [ "$need_deactivate" -eq 1 ]; then
            conda deactivate
        fi
        return 0
    fi
    
    # 方法2: 直接检查命令（如果环境已激活）
    if command -v "$tool_name" &>/dev/null; then
        if [ "$need_deactivate" -eq 1 ]; then
            conda deactivate
        fi
        return 0
    fi
    
    if [ "$need_deactivate" -eq 1 ]; then
        conda deactivate
    fi
    return 1
}

check_tool_version() {
    # 检查工具版本（通用函数）
    # 参数: $1 = 工具名, $2 = 版本要求（如 "3.18"），$3 = "no_activate"
    local tool_name="$1"
    local version_requirement="$2"
    local need_deactivate=0
    
    if [[ "$3" != "no_activate" ]]; then
        source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
        need_deactivate=1
    fi
    
    # 获取工具版本
    local version_output=""
    if conda run -n "${ENV_NAME}" "$tool_name" --version &>/dev/null; then
        version_output=$(conda run -n "${ENV_NAME}" "$tool_name" --version 2>/dev/null)
    elif command -v "$tool_name" &>/dev/null; then
        version_output=$("$tool_name" --version 2>/dev/null)
    else
        if [ "$need_deactivate" -eq 1 ]; then
            conda deactivate
        fi
        return 1
    fi
    
    if [ -z "$version_output" ]; then
        if [ "$need_deactivate" -eq 1 ]; then
            conda deactivate
        fi
        return 1
    fi
    
    # 提取版本号（尝试多种格式）
    local version=$(echo "$version_output" | head -n1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1)
    
    if [ -z "$version" ]; then
        if [ "$need_deactivate" -eq 1 ]; then
            conda deactivate
        fi
        return 1
    fi
    
    # 如果有版本要求，进行比较
    if [ -n "$version_requirement" ]; then
        local major=$(echo "$version" | cut -d. -f1)
        local minor=$(echo "$version" | cut -d. -f2)
        local version_num=$((major * 100 + minor))
        
        local req_major=$(echo "$version_requirement" | cut -d. -f1)
        local req_minor=$(echo "$version_requirement" | cut -d. -f2)
        local req_num=$((req_major * 100 + req_minor))
        
        if [ "$version_num" -ge "$req_num" ]; then
            if [ "$need_deactivate" -eq 1 ]; then
                conda deactivate
            fi
            echo "$version"
            return 0
        else
            if [ "$need_deactivate" -eq 1 ]; then
                conda deactivate
            fi
            echo "$version"
            return 1
        fi
    fi
    
    if [ "$need_deactivate" -eq 1 ]; then
        conda deactivate
    fi
    echo "$version"
    return 0
}

check_environment_tools() {
    local missing=()
    local not_executable=()
    
    # 如果环境未激活，则激活
    local need_deactivate=0
    if [[ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]]; then
        source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
        need_deactivate=1
    fi
    
    for tool in "${TOOLS[@]}"; do
        # 提取工具名（去除版本约束）
        tool_name=$(echo "$tool" | sed 's/[<>=].*//')
        
        # 检查是否已安装（如果在当前环境中，直接使用 conda list，否则使用 -n 参数）
        if [[ "$CONDA_DEFAULT_ENV" == "${ENV_NAME}" ]]; then
            # 在当前环境中，直接使用 conda list
            if ! conda list "$tool_name" &>/dev/null; then
                missing+=($tool)
                continue
            fi
        else
            # 不在当前环境中，使用 -n 参数
            if ! conda list -n "${ENV_NAME}" "$tool_name" &>/dev/null; then
                missing+=($tool)
                continue
            fi
        fi
        
        # 检查是否可执行
        if ! check_tool_executable "$tool_name" "no_activate"; then
            not_executable+=($tool)
        fi
    done
    
    # 如果之前激活了环境，现在退出
    if [ "$need_deactivate" -eq 1 ]; then
        conda deactivate
    fi
    
    if [ ${#missing[@]} -eq 0 ] && [ ${#not_executable[@]} -eq 0 ]; then
        return 0  # 工具完整且可执行
    else
        return 1  # 工具有缺失或不可执行
    fi
}

check_cmake_version() {
    # [已弃用] 此函数已被通用的 check_tool_version 函数替代
    # 保留此函数仅作为备用
    # 检查 cmake 是否已安装且版本 >= 3.18
    # 参数: $1 = "no_activate" 表示环境已激活，不需要再次激活
    
    # 首先检查 cmake 是否已安装（如果在当前环境中，直接使用 conda list，否则使用 -n 参数）
    local cmake_check_cmd
    if [[ "$CONDA_DEFAULT_ENV" == "${ENV_NAME}" ]]; then
        cmake_check_cmd="conda list cmake"
    else
        cmake_check_cmd="conda list -n \"${ENV_NAME}\" cmake"
    fi
    
    if ! eval "$cmake_check_cmd" &>/dev/null; then
        echo -e "${RED}  ✗ cmake 未安装${NC}"
        return 1
    fi
    
    # 使用 conda run 来运行 cmake，确保在正确的环境中
    local cmake_output
    if ! cmake_output=$(conda run -n "${ENV_NAME}" cmake --version 2>/dev/null); then
        # 如果 conda run 失败，尝试直接激活环境
        local need_deactivate=0
        if [[ "$1" != "no_activate" ]]; then
            source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
            need_deactivate=1
        fi
        
        # 刷新 PATH（conda install 后可能需要）
        hash -r 2>/dev/null || true
        
        if ! command -v cmake &>/dev/null; then
            echo -e "${RED}  ✗ cmake 未找到（可能需要在环境中手动激活）${NC}"
            echo -e "${YELLOW}  提示: 请运行 'conda activate ${ENV_NAME}' 后手动检查 cmake 版本${NC}"
            if [ "$need_deactivate" -eq 1 ]; then
                conda deactivate
            fi
            return 1
        fi
        
        cmake_output=$(cmake --version 2>/dev/null)
        if [ "$need_deactivate" -eq 1 ]; then
            conda deactivate
        fi
    fi
    
    # 获取 cmake 版本号
    local cmake_version=$(echo "$cmake_output" | head -n1 | awk '{print $3}')
    if [ -z "$cmake_version" ]; then
        echo -e "${RED}  ✗ 无法获取 cmake 版本信息${NC}"
        return 1
    fi
    
    local major=$(echo "$cmake_version" | cut -d. -f1)
    local minor=$(echo "$cmake_version" | cut -d. -f2)
    local version_num=$((major * 100 + minor))
    local required_num=318  # 3.18 = 318
    
    if [ "$version_num" -ge "$required_num" ]; then
        echo -e "${GREEN}  ✓ cmake 版本检查通过: ${cmake_version} (>= 3.18)${NC}"
        return 0
    else
        echo -e "${RED}  ✗ cmake 版本过低: ${cmake_version} (需要 >= 3.18)${NC}"
        echo -e "${YELLOW}  请尝试手动升级: conda install -n ${ENV_NAME} -c conda-forge 'cmake>=3.18'${NC}"
        return 1
    fi
}

detect_current_env() {
    # 检测当前是否在 conda 环境中
    if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
        echo -e "${GREEN}>>> 检测到当前已在 conda 环境中: ${CONDA_DEFAULT_ENV}${NC}"
        ENV_NAME="$CONDA_DEFAULT_ENV"
        return 0  # 在环境中
    else
        return 1  # 不在环境中
    fi
}

create_env() {
    # 如果当前已在 conda 环境中，直接使用当前环境
    if detect_current_env; then
        echo -e "${GREEN}>>> 使用当前环境: ${ENV_NAME}${NC}"
        # 检查当前环境的工具是否完整
        if check_environment_tools; then
            echo -e "${GREEN}>>> 当前环境中工具已完整${NC}"
            return
        else
            echo -e "${YELLOW}>>> 当前环境中工具不完整，将在当前环境中安装缺失的工具${NC}"
            return
        fi
    fi
    
    # 如果不在环境中，检查默认环境是否存在
    if ! conda env list | grep -q "${ENV_NAME}"; then
        echo -e "${YELLOW}>>> 创建新环境: ${ENV_NAME}${NC}"
        conda create -y -n "${ENV_NAME}" || {
            echo -e "${RED}环境创建失败!${NC}"
            exit 1
        }
        # 创建环境后立即激活
        echo -e "${YELLOW}>>> 激活环境: ${ENV_NAME}${NC}"
        source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
        return
    fi

    echo -e "${YELLOW}>>> 环境 ${ENV_NAME} 已存在${NC}"
    # 激活已存在的环境
    echo -e "${YELLOW}>>> 激活环境: ${ENV_NAME}${NC}"
    source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
    
    if check_environment_tools; then
        echo -e "${GREEN}>>> 环境中工具已完整${NC}"
        return
    else
        echo -e "${RED}>>> 环境中工具不完整${NC}"
        echo -e "${YELLOW}>>> 将在当前环境中安装缺失的工具${NC}"
    fi
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
    
    # 确保环境已激活（如果未激活则激活）
    if [[ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]]; then
        echo -e "${YELLOW}>>> 激活环境: ${ENV_NAME}${NC}"
        source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
    else
        echo -e "${GREEN}>>> 环境 ${ENV_NAME} 已激活${NC}"
    fi
    
    for tool in "${TOOLS[@]}"; do
        # 提取工具名（去除版本约束）
        tool_name=$(echo "$tool" | sed 's/[<>=].*//')
        
        # 检查工具是否已安装 (如果在当前环境中，直接使用 conda list，否则使用 -n 参数)
        local list_cmd
        if [[ "$CONDA_DEFAULT_ENV" == "${ENV_NAME}" ]]; then
            list_cmd="conda list \"$tool_name\""
        else
            list_cmd="conda list -n \"${ENV_NAME}\" \"$tool_name\""
        fi
        
        if eval "$list_cmd" &>/dev/null; then
            # 进一步检查工具是否可执行
            if check_tool_executable "$tool_name" "no_activate"; then
                echo -e "${BLUE}▸ ${tool_name} 已安装且可执行，跳过更新...${NC}"
                installed+=("$tool")
                continue
            else
                echo -e "${YELLOW}▸ ${tool_name} 已安装但不可执行，尝试重新安装...${NC}"
            fi
        fi
        
        echo -e "${BLUE}▸ 正在安装 ${tool}...${NC}"
        
        # cmake 需要从 conda-forge 安装以确保版本>=3.18
        if [[ "$tool_name" == "cmake" ]]; then
            if conda install -y -n "${ENV_NAME}" -c conda-forge "${tool}"; then
                installed+=("$tool")
                echo -e "${GREEN}  ✓ ${tool} 安装成功${NC}"
                # 等待一下，确保环境更新
                sleep 1
                # 检查 cmake 是否可执行
                echo -e "${BLUE}  ▸ 检查 cmake 是否可执行...${NC}"
                if check_tool_executable "$tool_name" "no_activate"; then
                    echo -e "${GREEN}  ✓ cmake 可执行${NC}"
                    # 检查 cmake 版本是否符合要求
                    echo -e "${BLUE}  ▸ 检查 cmake 版本...${NC}"
                    local cmake_version=$(check_tool_version "$tool_name" "3.18" "no_activate")
                    if [ $? -eq 0 ]; then
                        echo -e "${GREEN}  ✓ cmake 版本验证通过: ${cmake_version} (>= 3.18)${NC}"
                    else
                        echo -e "${YELLOW}  ⚠ cmake 版本可能不符合要求: ${cmake_version} (需要 >= 3.18)${NC}"
                    fi
                else
                    echo -e "${YELLOW}  ⚠ cmake 已安装但不可执行（可能需要重新激活环境）${NC}"
                    echo -e "${YELLOW}  提示: 运行 'conda activate ${ENV_NAME}' 后执行 'cmake --version' 验证${NC}"
                fi
            else
                failed+=("$tool")
                echo -e "${RED}  ✗ ${tool} 安装失败${NC}"
            fi
        else
            if conda install -y -n "${ENV_NAME}" "${tool}"; then
                installed+=("$tool")
                echo -e "${GREEN}  ✓ ${tool} 安装成功${NC}"
                # 等待一下，确保环境更新
                sleep 1
                # 检查工具是否可执行
                echo -e "${BLUE}  ▸ 检查 ${tool_name} 是否可执行...${NC}"
                if check_tool_executable "$tool_name" "no_activate"; then
                    # 尝试获取版本信息
                    local tool_version=$(check_tool_version "$tool_name" "" "no_activate" 2>/dev/null)
                    if [ -n "$tool_version" ]; then
                        echo -e "${GREEN}  ✓ ${tool_name} 可执行，版本: ${tool_version}${NC}"
                    else
                        echo -e "${GREEN}  ✓ ${tool_name} 可执行${NC}"
                    fi
                else
                    echo -e "${YELLOW}  ⚠ ${tool_name} 已安装但不可执行（可能需要重新激活环境）${NC}"
                fi
            else
                failed+=("$tool")
                echo -e "${RED}  ✗ ${tool} 安装失败${NC}"
            fi
        fi
        
        # 保存安装进度
        if [ ${#installed[@]} -gt 0 ]; then
            echo "installed=(${installed[@]})" > "$PROGRESS_FILE"
            echo "failed=(${failed[@]})" >> "$PROGRESS_FILE"
        fi
    done
    
    # 显示汇总报告
    echo -e "\n${GREEN}======== 安装报告 ========${NC}"
    echo -e "成功: ${GREEN}${#installed[@]}${NC} 个"
    echo -e "失败: ${RED}${#failed[@]}${NC} 个"
    
    # 特别检查 cmake 版本（如果已安装）
    # 如果在当前环境中，直接使用 conda list，否则使用 -n 参数
    local cmake_list_cmd
    if [[ "$CONDA_DEFAULT_ENV" == "${ENV_NAME}" ]]; then
        cmake_list_cmd="conda list cmake"
    else
        cmake_list_cmd="conda list -n \"${ENV_NAME}\" cmake"
    fi
    
    if eval "$cmake_list_cmd" &>/dev/null; then
        echo -e "\n${BLUE}======== cmake 版本检查 ========${NC}"
        if check_tool_executable "cmake"; then
            local cmake_version=$(check_tool_version "cmake" "3.18")
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}cmake 版本符合要求: ${cmake_version} (>= 3.18)，可以正常使用${NC}"
            else
                echo -e "${YELLOW}cmake 版本可能不符合要求: ${cmake_version} (需要 >= 3.18)${NC}"
                echo -e "${YELLOW}请手动验证: conda activate ${ENV_NAME} && cmake --version${NC}"
            fi
        else
            echo -e "${YELLOW}cmake 已安装但不可执行${NC}"
            echo -e "${YELLOW}请手动验证: conda activate ${ENV_NAME} && cmake --version${NC}"
        fi
    fi
    
    if [ ${#failed[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}以下工具安装失败:${NC}"
        printf ' - %s\n' "${failed[@]}"
        echo -e "\n${YELLOW}您可以尝试手动安装:"
        echo -e "conda install -n ${ENV_NAME} 工具名${NC}"
    fi
    
    # 注意：不在这里退出环境，保持环境激活状态以便用户使用
    # 如果需要退出环境，用户可以手动执行 conda deactivate
    
    # 清理进度文件
    if [ -f "$PROGRESS_FILE" ]; then
        rm -f "$PROGRESS_FILE"
    fi
}

check_cmake_for_python() {
    # 检查 cmake 是否已安装（用于 Python 包安装前的检查）
    local cmake_list_cmd
    if [[ "$CONDA_DEFAULT_ENV" == "${ENV_NAME}" ]]; then
        cmake_list_cmd="conda list cmake"
    else
        cmake_list_cmd="conda list -n \"${ENV_NAME}\" cmake"
    fi
    
    if ! eval "$cmake_list_cmd" &>/dev/null; then
        return 1
    fi
    
    # 检查 cmake 版本（确保环境已激活）
    if [[ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]]; then
        source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
    fi
    
    local cmake_version=$(check_tool_version "cmake" "3.18" "no_activate" 2>/dev/null)
    local version_check=$?
    
    if [[ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]]; then
        conda deactivate 2>/dev/null || true
    fi
    
    if [ $version_check -eq 0 ] && [ -n "$cmake_version" ]; then
        return 0
    fi
    return 1
}

install_python_package() {
    # 安装 assoG2P Python 包
    # 参数: $1 = whl 文件路径
    local whl_file="$1"
    
    if [ ! -f "$whl_file" ]; then
        echo -e "${RED}错误: whl 文件不存在: ${whl_file}${NC}"
        return 1
    fi
    
    echo -e "\n${YELLOW}>>> 开始安装 assoG2P Python 包${NC}"
    
    # 检查 cmake 是否已安装（xgboost 需要）
    if ! check_cmake_for_python; then
        echo -e "${RED}错误: cmake 未安装或版本不符合要求（需要 >= 3.18）${NC}"
        echo -e "${YELLOW}请先运行 ./tools.sh 安装 cmake，然后再安装 whl 包${NC}"
        return 1
    fi
    
    # 确保环境已激活
    if [[ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]]; then
        echo -e "${YELLOW}>>> 激活环境: ${ENV_NAME}${NC}"
        source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
    fi
    
    echo -e "${BLUE}>>> 安装 whl 包: ${whl_file}${NC}"
    
    # 尝试使用预编译的依赖包（避免编译问题）
    echo -e "${BLUE}>>> 预安装依赖包（使用预编译版本）...${NC}"
    pip install --prefer-binary xgboost lightgbm catboost 2>/dev/null || true
    
    # 安装 whl 包
    if pip install "$whl_file"; then
        echo -e "${GREEN}>>> assoG2P Python 包安装成功${NC}"
        
        # 验证安装
        if command -v association &>/dev/null; then
            local version=$(association --version 2>/dev/null || echo "未知")
            echo -e "${GREEN}>>> 验证安装: association 命令可用，版本: ${version}${NC}"
        else
            echo -e "${YELLOW}>>> 警告: association 命令未找到，可能需要重新激活环境${NC}"
        fi
        return 0
    else
        echo -e "${RED}>>> assoG2P Python 包安装失败${NC}"
        echo -e "${YELLOW}提示: 如果遇到 CMake 错误，请确保 cmake>=3.18 已安装${NC}"
        return 1
    fi
}

show_usage() {
    echo -e "${GREEN}
环境状态:
  当前环境: ${CONDA_DEFAULT_ENV:-未激活}
  环境名称: ${ENV_NAME}
  
使用方法:
  # 如果环境未激活，使用以下命令激活:
  source ${CONDA_PATH}/bin/activate ${ENV_NAME}
  
  # 或者:
  conda activate ${ENV_NAME}

常用命令:
  conda list              查看已安装工具
  conda update --all      更新所有工具
  conda remove 工具名     卸载特定工具
  conda deactivate        退出当前环境

安装 assoG2P Python 包:
  # 方法1: 使用 tools.sh 安装（推荐）
  ./tools.sh --whl /path/to/assoG2P-1.0.0-py3-none-any.whl
  
  # 方法2: 手动安装
  conda activate ${ENV_NAME}
  pip install /path/to/assoG2P-1.0.0-py3-none-any.whl

环境管理:
  conda env export > environment.yml  导出环境配置
  conda env create -f environment.yml 从配置恢复环境${NC}"
}

# =============== 命令行参数解析 ===============
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --whl)
                WHL_FILE="$2"
                if [ ! -f "$WHL_FILE" ]; then
                    echo -e "${RED}错误: whl 文件不存在: ${WHL_FILE}${NC}"
                    exit 1
                fi
                shift 2
                ;;
            --install-python-deps)
                INSTALL_PYTHON_DEPS=true
                shift
                ;;
            -h|--help)
                echo -e "${GREEN}用法: $0 [选项]${NC}"
                echo -e "${GREEN}选项:${NC}"
                echo -e "  --whl <文件路径>          安装指定的 whl 文件（assoG2P Python 包）"
                echo -e "  --install-python-deps     提示安装 Python 依赖（需要配合 --whl 使用）"
                echo -e "  -h, --help                显示此帮助信息"
                exit 0
                ;;
            *)
                echo -e "${RED}未知选项: $1${NC}"
                echo -e "${YELLOW}使用 -h 或 --help 查看帮助信息${NC}"
                exit 1
                ;;
        esac
    done
}

# =============== 主流程 ===============
main() {
    # 解析命令行参数
    parse_args "$@"
    
    init_colors
    
    # 首先检测是否已在 conda 环境中（在初始化 conda 之前检测）
    if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
        ENV_NAME="$CONDA_DEFAULT_ENV"
        echo -e "${GREEN}>>> 检测到当前已在 conda 环境中: ${ENV_NAME}${NC}"
        echo -e "${GREEN}>>> 将在当前环境中安装工具${NC}\n"
    fi
    
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
    
    setup_conda_channels
    
    # 检测并设置环境（如果在当前环境中，使用当前环境；否则创建/激活默认环境）
    create_env
    
    # 确保环境已激活（双重保险）
    if [[ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]]; then
        echo -e "${YELLOW}>>> 激活环境: ${ENV_NAME}${NC}"
        source "${CONDA_PATH}/bin/activate" "${ENV_NAME}"
    else
        echo -e "${GREEN}>>> 当前环境: ${ENV_NAME} 已激活${NC}"
    fi
    
    # 在激活的环境中安装工具
    install_tools
    
    # 如果指定了 whl 文件，安装 Python 包
    if [ -n "$WHL_FILE" ] && [ -f "$WHL_FILE" ]; then
        install_python_package "$WHL_FILE"
    elif [ "$INSTALL_PYTHON_DEPS" = true ]; then
        echo -e "${YELLOW}>>> 提示: 要安装 assoG2P Python 包，请提供 whl 文件路径${NC}"
        echo -e "${YELLOW}>>> 使用方法: ./tools.sh --whl /path/to/assoG2P-1.0.0-py3-none-any.whl${NC}"
    fi
    
    echo -e "\n${GREEN}=== 安装完成! ===${NC}"
    echo -e "${GREEN}当前环境: ${CONDA_DEFAULT_ENV}${NC}"
    show_usage
}

main "$@"