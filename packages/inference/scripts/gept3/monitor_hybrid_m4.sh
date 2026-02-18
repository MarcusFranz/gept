#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${2:-${RUN_NAME:-gept3_hybrid_m4_full_s128_t48_n3k_e12_b2_ga4}}"
LOG="${1:-/Users/marcus/code/gept/packages/inference/logs/${RUN_NAME}.log}"
TOTAL_EPOCHS="${3:-${TOTAL_EPOCHS:-12}}"
WATCHDOG_LOG="${WATCHDOG_LOG:-/Users/marcus/code/gept/packages/inference/logs/${RUN_NAME}_watchdog.log}"

pid="$(pgrep -f "train_gept3_hybrid.py.*${RUN_NAME}" | head -n1 || true)"
echo "log: ${LOG}"
if [[ -n "${pid}" ]]; then
  echo "status: running (pid=${pid})"
  ps -o pid,%cpu,%mem,rss,vsz,etime,stat,command -p "${pid}" | sed 1d
else
  echo "status: not running"
fi

watchdog_pid="$(pgrep -f "watch_hybrid_m4.sh" | tail -n1 || true)"
if [[ -n "${watchdog_pid}" ]]; then
  echo "watchdog: running (pid=${watchdog_pid})"
else
  echo "watchdog: not running"
fi

if [[ ! -f "${LOG}" ]]; then
  echo "log file not found yet."
  exit 0
fi

echo
echo "latest log lines:"
tail -n 20 "${LOG}"

if [[ -f "${WATCHDOG_LOG}" ]]; then
  echo
  echo "watchdog log tail:"
  tail -n 8 "${WATCHDOG_LOG}"
fi

last_start_line="$(grep -n "starting hybrid full run" "${LOG}" | tail -n1 | cut -d: -f1 || true)"
if [[ -z "${last_start_line}" ]]; then
  last_start_line=1
fi

last_progress="$(awk -v s="${last_start_line}" 'NR >= s' "${LOG}" | grep -E "epoch=[0-9]+ batch=[0-9]+/[0-9]+" | tail -n1 || true)"
if [[ -z "${last_progress}" ]]; then
  exit 0
fi

epoch="$(echo "${last_progress}" | sed -E 's/.*epoch=([0-9]+).*/\1/')"
batch_cur="$(echo "${last_progress}" | sed -E 's/.*batch=([0-9]+)\/([0-9]+).*/\1/')"
batch_total="$(echo "${last_progress}" | sed -E 's/.*batch=([0-9]+)\/([0-9]+).*/\2/')"
elapsed_s="$(echo "${last_progress}" | sed -E 's/.*elapsed_s=([0-9]+(\.[0-9]+)?).*/\1/')"

if [[ -z "${epoch}" || -z "${batch_cur}" || -z "${batch_total}" || -z "${elapsed_s}" ]]; then
  exit 0
fi

if [[ "${batch_cur}" -gt 0 ]]; then
  per_batch_s="$(awk -v e="${elapsed_s}" -v b="${batch_cur}" 'BEGIN{printf "%.4f", e/b}')"
else
  per_batch_s="0"
fi

remain_epoch_batches=$((batch_total - batch_cur))
remain_total_batches=$((remain_epoch_batches + (TOTAL_EPOCHS - epoch) * batch_total))

eta_epoch_s="$(awk -v p="${per_batch_s}" -v r="${remain_epoch_batches}" 'BEGIN{printf "%.0f", p*r}')"
eta_total_s="$(awk -v p="${per_batch_s}" -v r="${remain_total_batches}" 'BEGIN{printf "%.0f", p*r}')"

echo
echo "progress:"
echo "epoch ${epoch}/${TOTAL_EPOCHS}, batch ${batch_cur}/${batch_total}, sec_per_batch~${per_batch_s}"
echo "eta_epoch~${eta_epoch_s}s, eta_total~${eta_total_s}s (train-only rough estimate)"
