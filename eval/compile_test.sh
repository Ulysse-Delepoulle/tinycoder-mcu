#!/usr/bin/env bash
# Compile-check a single .c file for the STM32N6 (Cortex-M55).
#
# Exit codes (evaluate.py depends on these):
#   0 -> compiled cleanly
#   1 -> compilation failed
#   2 -> arm-none-eabi-gcc not installed (SOFT skip, not a real failure)
#
# We only run a syntax/compile check (-c, output discarded). We are NOT linking
# a full firmware image — no startup code, linker script, or libc is required
# just to verify the generated C is well-formed for the target.

set -u

C_FILE="${1:-}"
if [[ -z "${C_FILE}" ]]; then
  echo "usage: compile_test.sh <file.c>" >&2
  exit 1
fi

# Soft-fail if the ARM cross-compiler is missing (e.g. on Kaggle, which has no
# arm-none-eabi toolchain). evaluate.py treats exit code 2 as "skip", not "fail".
if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "arm-none-eabi-gcc not found — skipping compilation test"
  exit 2
fi

# -mcpu=cortex-m55        : the STM32N6 core
# -mfpu=fpv5-d16          : its FPU (single/double precision, 16 D-registers)
# -mfloat-abi=hard        : use FPU registers for float args (typical for M55)
# -c -o /dev/null         : compile only, throw away the object file
# -Wall                   : surface obvious problems as warnings (not fatal here)
arm-none-eabi-gcc \
  -mcpu=cortex-m55 \
  -mfpu=fpv5-d16 \
  -mfloat-abi=hard \
  -Wall \
  -c "${C_FILE}" \
  -o /dev/null

# Propagate gcc's exit status: 0 on success, non-zero (->1) on compile error.
if [[ $? -eq 0 ]]; then
  exit 0
else
  exit 1
fi
