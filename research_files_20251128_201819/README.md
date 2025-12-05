# Research Files for Keras 3.x → PyTorch MHA Conversion Issue

## Contents

- **RESEARCH_CONTEXT.md** - Comprehensive research document with all context
- **RESEARCH_QUICK_REFERENCE.md** - Quick reference guide
- **src/** - Model definitions and supporting code
- **scripts/** - Conversion and test scripts
- **saved_models/** - Reference outputs and data

## Problem Summary

Policy outputs differ massively (max diff 10-13, 0/10 top move overlap) while value outputs are close (max diff 0.001), suggesting upstream MHA computation issue.

## Key Files

1. `src/transformer_model.py` - Keras model (lines 28-42 for MHA)
2. `src/transformer_model_pytorch.py` - PyTorch model (lines 81-140 for MHA)
3. `scripts/convert_keras_to_pytorch.py` - Conversion code (lines 175-242)
4. `saved_models/keras_outputs.json` - Reference outputs

See RESEARCH_CONTEXT.md for full details.
