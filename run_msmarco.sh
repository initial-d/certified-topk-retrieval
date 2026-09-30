#!/bin/sh
# Full MS MARCO pipeline; waits for the chunked encoding (89 chunks) to finish.
cd "$(dirname "$0")"
PY=../.venv/bin/python
until [ "$(wc -l < ../cache/msmarco.bge.docs.f16.done 2>/dev/null)" -ge 89 ]; do sleep 60; done
unset MS_LIMIT
$PY -m credal.exp_msmarco passA || exit 1
for q in pq32 pq96 sq4 bin; do
  $PY -m credal.exp_msmarco quant $q || exit 1
  $PY -m credal.exp_msmarco passB $q || exit 1
  $PY -m credal.exp_msmarco passC $q || exit 1
done
$PY -m credal.exp_msmarco report
echo MSMARCO-PIPELINE-DONE
