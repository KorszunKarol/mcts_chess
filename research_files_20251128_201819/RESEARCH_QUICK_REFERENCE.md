# Quick Reference for Deep Research

## Problem in One Sentence
Policy outputs differ massively (max diff 10-13, 0/10 top move overlap) while value outputs are close (max diff 0.001), suggesting upstream MHA computation issue.

## Key Files to Include

### Must Include:
1. `RESEARCH_CONTEXT.md` - This comprehensive document
2. `src/transformer_model.py` - Keras model (lines 28-42 for MHA)
3. `src/transformer_model_pytorch.py` - PyTorch model (lines 81-140 for MHA)
4. `scripts/convert_keras_to_pytorch.py` - Conversion code (lines 175-242)
5. `saved_models/keras_outputs.json` - Keras reference outputs

### Supporting Files:
6. `scripts/analyze_policy_differences.py` - Detailed difference analysis
7. `scripts/test_mha_conversion.py` - Q/K/V verification
8. `src/weights/best_model.keras` - Original model

## Key Data Points

### Output Differences:
- Value max diff: 0.001 ✓
- Policy max diff: 10-13 ✗
- Policy mean diff: 0.8-1.0 ✗
- Top 10 overlap: 0/10 ✗

### Weight Conversion:
- Coverage: 100% (135/135) ✓
- Q/K/V: Mathematically verified ✓
- Output projection: Matches document code ✓

## Critical Research Questions

1. How does Keras 3.x MHA compute attention vs PyTorch?
2. Are head outputs concatenated or summed?
3. Is there a scaling/normalization difference?
4. Are Q/K/V weights in the correct order?

## Test Results

Q/K/V conversion test: 0.000000 difference ✓
Output projection: Matches document exactly ✓
Value head: Works perfectly ✓
Policy head: Completely wrong ✗

