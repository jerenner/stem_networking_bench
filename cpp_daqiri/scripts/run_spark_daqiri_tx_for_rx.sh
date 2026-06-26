#!/usr/bin/env bash
# TX side for the TX+RX Spark gate. Uses the stem_daqiri:tx-rx container so it
# pairs with stem_daqiri_rx running on spark-stacked-02. Forwards all args to
# run_spark_daqiri_tx.sh.
exec "$(dirname "$0")/run_spark_daqiri_tx.sh" --image stem_daqiri:tx-rx "$@"
