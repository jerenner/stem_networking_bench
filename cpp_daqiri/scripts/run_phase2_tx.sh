#!/usr/bin/env bash
# Phase 2 runtime gate -- TX side. Same as Phase 1 TX but uses the
# stem_daqiri:phase2 container so it pairs with stem_daqiri_rx running on
# spark-stacked-02. Forwards all args to run_phase1_tx.sh.
exec "$(dirname "$0")/run_phase1_tx.sh" --image stem_daqiri:phase2 "$@"
