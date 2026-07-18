#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# 실행 인자 검사
# 사용법: ./train_and_test_all_model_repeatedly.sh <반복 횟수>
# 예시:   ./train_and_test_all_model_repeatedly.sh 10
# ============================================================


if [[ $# -ne 1 ]]; then
    echo "사용법: $0 <반복 횟수>"
    echo "예시:   $0 10"
    exit 1
fi

repeatCount="$1"

# 1 이상의 정수인지 검사
if [[ ! "$repeatCount" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] 반복 횟수는 1 이상의 정수여야 합니다."
    echo "입력값: ${repeatCount}"
    exit 1
fi


# ============================================================
# 기본 경로 및 실행 시작 시각 설정
# ============================================================


# 이 스크립트를 실행한 현재 디렉터리
workDir="$(pwd)"

# 스크립트 실행 시작 시각
startTimestamp="$(date '+%F_%H-%M-%S')"

# 전체 실행 시간 측정을 위한 시작 시간
scriptStartEpoch="$(date +%s)"
scriptStartTime="$(date '+%F %T')"

# 전체 터미널 출력을 저장할 로그 파일
cmdLogFile_Dir="${workDir}/logs"
cmdLogFile="${workDir}/logs/train_${startTimestamp}.log"

# 이번 실행의 결과를 모을 디렉터리
resultsDir_0="${workDir}/results"
resultsDir="${workDir}/results/results_${startTimestamp}"

# experiment.py 실행 후 생성되는 디렉터리
experimentsDir="${workDir}/experiments"


# ============================================================
# 터미널 출력 로깅 설정
# ============================================================


mkdir -p "$cmdLogFile_Dir"
exec > >(stdbuf -oL -eL tee -a "$cmdLogFile") 2>&1

# Python의 출력 버퍼링 방지
export PYTHONUNBUFFERED=1


echo "============================================================"
echo "Repeated experiment script started"
echo "Working directory : ${workDir}"
echo "Repeat count      : ${repeatCount}"
echo "Results directory : ${resultsDir}"
echo "Log file          : ${cmdLogFile}"
echo "Start time        : ${scriptStartTime}"
echo "============================================================"


# ============================================================
# 새로운 results_날짜시간 디렉터리 생성
# ============================================================


if [[ -e "$resultsDir" ]]; then
    echo "[ERROR] 같은 이름의 결과 경로가 이미 존재합니다:"
    echo "        ${resultsDir}"
    echo "1초 후 다시 실행하거나 기존 경로를 확인해 주세요."
    exit 1
fi

mkdir -p "$resultsDir_0"
mkdir -- "$resultsDir"

echo "[Initialization] Created results directory:"
echo "                 ${resultsDir}"


# ============================================================
# 이전 experiments 디렉터리 존재 여부 확인
# ============================================================


# 이전 실행에서 생성된 experiments 디렉터리를 이번 실행의 결과로
# 잘못 이동하는 것을 막기 위해 자동 삭제하지 않고 실행을 중단한다.
if [[ -e "$experimentsDir" ]]; then
    echo
    echo "[ERROR] 실험을 시작하기 전에 experiments 경로가 이미 존재합니다:"
    echo "        ${experimentsDir}"
    echo
    echo "기존 실험 결과일 수 있으므로 자동으로 삭제하지 않습니다."
    echo "해당 경로를 직접 이동하거나 삭제한 후 다시 실행해 주세요."
    exit 1
fi


# ============================================================
# experiment.py 반복 실행
# ============================================================


for ((i = 1; i <= repeatCount; i++)); do
    # 전체 반복 횟수의 자릿수에 따라 최소 두 자리로 표시
    # 예: 1 -> 01, 10 -> 10, 100 -> 100
    printf -v repeatNumber "%02d" "$i"

    destinationDir="${resultsDir}/experiments_repeat_${repeatNumber}"

    echo
    echo "============================================================"
    echo "[Repeat ${repeatNumber}/${repeatCount}] Experiment started"
    echo "Start time: $(date '+%F %T')"
    echo "Command   : python -m test.experiment"
    echo "============================================================"

    # 혹시 직전 반복에서 experiments 디렉터리가 남아 있는지 검사
    if [[ -e "$experimentsDir" ]]; then
        echo "[ERROR] 실험 실행 전에 experiments 경로가 이미 존재합니다:"
        echo "        ${experimentsDir}"
        echo "반복 실행을 중단합니다."
        exit 1
    fi

    # 실험 실행
    if python -m test.experiment; then
        echo
        echo "[Repeat ${repeatNumber}/${repeatCount}] Python process completed."
    else
        exitCode=$?

        echo
        echo "[ERROR] Repeat ${repeatNumber}/${repeatCount} failed."
        echo "Exit code: ${exitCode}"
        echo "이후 반복 실험은 실행하지 않습니다."
        echo
        echo "현재까지 완료된 결과는 다음 경로에 보존되어 있습니다:"
        echo "${resultsDir}"

        exit "$exitCode"
    fi

    # experiment.py가 experiments 디렉터리를 생성했는지 확인
    if [[ ! -d "$experimentsDir" ]]; then
        echo
        echo "[ERROR] Python 명령은 정상 종료되었지만 experiments 디렉터리가 생성되지 않았습니다:"
        echo "        ${experimentsDir}"
        echo "이후 반복 실험은 실행하지 않습니다."
        exit 1
    fi

    # 동일한 이름의 이동 대상이 이미 존재하는지 확인
    if [[ -e "$destinationDir" ]]; then
        echo
        echo "[ERROR] 이동할 대상 경로가 이미 존재합니다:"
        echo "        ${destinationDir}"
        echo "기존 결과를 보호하기 위해 실행을 중단합니다."
        exit 1
    fi

    # experiments 디렉터리를 results_날짜시간 안으로 이동하면서 이름 변경
    mv -- "$experimentsDir" "$destinationDir"

    echo "[Repeat ${repeatNumber}/${repeatCount}] Results moved:"
    echo "  ${experimentsDir}"
    echo "  -> ${destinationDir}"
    echo "End time: $(date '+%F %T')"
done


# ============================================================
# 전체 반복 완료
# ============================================================


# 전체 실행 종료 시간과 총 소요 시간 계산
scriptEndEpoch="$(date +%s)"
scriptEndTime="$(date '+%F %T')"

elapsedSeconds=$((scriptEndEpoch - scriptStartEpoch))

elapsedDays=$((elapsedSeconds / 86400))
elapsedHours=$(((elapsedSeconds % 86400) / 3600))
elapsedMinutes=$(((elapsedSeconds % 3600) / 60))
elapsedRemainingSeconds=$((elapsedSeconds % 60))

if (( elapsedDays > 0 )); then
    printf -v elapsedTime "%d일 %02d시간 %02d분 %02d초" \
        "$elapsedDays" \
        "$elapsedHours" \
        "$elapsedMinutes" \
        "$elapsedRemainingSeconds"
else
    printf -v elapsedTime "%02d시간 %02d분 %02d초" \
        "$elapsedHours" \
        "$elapsedMinutes" \
        "$elapsedRemainingSeconds"
fi


echo
echo "============================================================"
echo "All experiments completed successfully."
echo "Completed repeats : ${repeatCount}"
echo "Results directory : ${resultsDir}"
echo "Log file          : ${cmdLogFile}"
echo "Start time        : ${scriptStartTime}"
echo "End time          : ${scriptEndTime}"
echo "Total elapsed time: ${elapsedTime}"
echo "============================================================"