#!/bin/bash
# Run the test set evaluation and save output
# Expected runtime: 2-3 hours on CPU (608 pairs, 6 bi-encoders + cross-encoder)
export USE_TF=0
export PYTHONUNBUFFERED=1
cd /Users/ceverson/.superset/worktrees/ai-helpdesk-agent/ceverson70/eval-update
python3 -m eval.compare_strategies \
    --test-set \
    --models minilm mpnet bge e5 minilm-ft mpnet-ft \
    --save eval/data/test_set_results.json \
    2>&1 | tee /tmp/test_set_eval.log
