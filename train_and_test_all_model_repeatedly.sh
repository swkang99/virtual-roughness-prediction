#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# Validate command-line arguments
# Usage: ./train_and_test_all_model_repeatedly.sh <number of repetitions>
# Example: ./train_and_test_all_model_repeatedly.sh 10
# ============================================================


# Use 1 as the default when no argument is provided
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [number of repetitions]"
    echo "Example: $0 10"
    exit 1
fi

repeatCount="${1:-1}"

# Check whether the value is an integer greater than or equal to 1
if [[ ! "$repeatCount" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] The number of repetitions must be an integer greater than or equal to 1."
    echo "Input value: ${repeatCount}"
    exit 1
fi


# ============================================================
# Configure basic paths and the execution start time
# ============================================================


# Current directory from which this script is executed
workDir="$(pwd)"

# Script execution start timestamp
startTimestamp="$(date '+%F_%H-%M-%S')"

# Start time for measuring the total execution time
scriptStartEpoch="$(date +%s)"
scriptStartTime="$(date '+%F %T')"

# Log file for storing all terminal output
cmdLogFile_Dir="${workDir}/logs"
cmdLogFile="${workDir}/logs/train_${startTimestamp}.log"

# Directory for collecting the results of this run
resultsDir_0="${workDir}/results"
resultsDir="${workDir}/results/results_${startTimestamp}"

# Directory created after experiment.py is executed
experimentsDir="${workDir}/experiments"


# ============================================================
# Configure terminal output logging
# ============================================================


mkdir -p "$cmdLogFile_Dir"
exec > >(stdbuf -oL -eL tee -a "$cmdLogFile") 2>&1

# Disable Python output buffering
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
# Create a new results_<date_time> directory
# ============================================================


if [[ -e "$resultsDir" ]]; then
    echo "[ERROR] A results path with the same name already exists:"
    echo "        ${resultsDir}"
    echo "Please run the script again after one second or check the existing path."
    exit 1
fi

mkdir -p "$resultsDir_0"
mkdir -- "$resultsDir"

echo "[Initialization] Created results directory:"
echo "                 ${resultsDir}"


# ============================================================
# Check whether a previous experiments directory exists
# ============================================================


# To prevent an experiments directory created by a previous run from being
# incorrectly moved as part of this run, stop execution instead of deleting it automatically.
if [[ -e "$experimentsDir" ]]; then
    echo
    echo "[ERROR] The experiments path already exists before starting the experiment:"
    echo "        ${experimentsDir}"
    echo
    echo "It may contain results from a previous experiment, so it will not be deleted automatically."
    echo "Please manually move or delete the path, and then run the script again."
    exit 1
fi


# ============================================================
# Repeatedly execute experiment.py
# ============================================================


for ((i = 1; i <= repeatCount; i++)); do
    # Display the repetition number using at least two digits,
    # depending on the number of digits in the total repetition count
    # Example: 1 -> 01, 10 -> 10, 100 -> 100
    printf -v repeatNumber "%02d" "$i"

    destinationDir="${resultsDir}/experiments_repeat_${repeatNumber}"

    echo
    echo "============================================================"
    echo "[Repeat ${repeatNumber}/${repeatCount}] Experiment started"
    echo "Start time: $(date '+%F %T')"
    echo "Command   : python -m test.experiment"
    echo "============================================================"

    # Check whether the experiments directory remains from the previous repetition
    if [[ -e "$experimentsDir" ]]; then
        echo "[ERROR] The experiments path already exists before executing the experiment:"
        echo "        ${experimentsDir}"
        echo "Stopping the repeated experiment execution."
        exit 1
    fi

    # Execute the experiment
    if python -m test.experiment; then
        echo
        echo "[Repeat ${repeatNumber}/${repeatCount}] Python process completed."
    else
        exitCode=$?

        echo
        echo "[ERROR] Repeat ${repeatNumber}/${repeatCount} failed."
        echo "Exit code: ${exitCode}"
        echo "No further experiment repetitions will be executed."
        echo
        echo "The results completed so far have been preserved in the following directory:"
        echo "${resultsDir}"

        exit "$exitCode"
    fi

    # Check whether experiment.py created the experiments directory
    if [[ ! -d "$experimentsDir" ]]; then
        echo
        echo "[ERROR] The Python command completed successfully, but the experiments directory was not created:"
        echo "        ${experimentsDir}"
        echo "No further experiment repetitions will be executed."
        exit 1
    fi

    # Check whether a destination with the same name already exists
    if [[ -e "$destinationDir" ]]; then
        echo
        echo "[ERROR] The destination path already exists:"
        echo "        ${destinationDir}"
        echo "Execution will be stopped to protect the existing results."
        exit 1
    fi

    # Move the experiments directory into the results_<date_time> directory and rename it
    mv -- "$experimentsDir" "$destinationDir"

    echo "[Repeat ${repeatNumber}/${repeatCount}] Results moved:"
    echo "  ${experimentsDir}"
    echo "  -> ${destinationDir}"
    echo "End time: $(date '+%F %T')"
done


# ============================================================
# All repetitions completed
# ============================================================


# Calculate the overall execution end time and total elapsed time
scriptEndEpoch="$(date +%s)"
scriptEndTime="$(date '+%F %T')"

elapsedSeconds=$((scriptEndEpoch - scriptStartEpoch))

elapsedDays=$((elapsedSeconds / 86400))
elapsedHours=$(((elapsedSeconds % 86400) / 3600))
elapsedMinutes=$(((elapsedSeconds % 3600) / 60))
elapsedRemainingSeconds=$((elapsedSeconds % 60))

if (( elapsedDays > 0 )); then
    printf -v elapsedTime "%d days %02d hours %02d minutes %02d seconds" \
        "$elapsedDays" \
        "$elapsedHours" \
        "$elapsedMinutes" \
        "$elapsedRemainingSeconds"
else
    printf -v elapsedTime "%02d hours %02d minutes %02d seconds" \
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