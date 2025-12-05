# Deep Research Context: Keras 3.x → PyTorch MHA Conversion Issue

## Executive Summary

**Problem:** Successfully converted a Hybrid CNN-Transformer chess model from Keras 3.x to PyTorch with 100% weight coverage, but policy outputs differ significantly (max diff ~12.4, 0/10 top move overlap) while value outputs are close (max diff ~0.001).

**Root Cause Hypothesis:** The issue is upstream in the MultiHeadAttention (MHA) computation, likely in Q/K/V weight conversion or attention computation, not in the output projection (which we've verified matches the documented solution).

---

## 1. Model Architecture

### Architecture Overview
- **Type:** Hybrid CNN-Transformer for chess position evaluation
- **Input:** `(8, 8, 34)` board representation
  - Keras format: `NHWC` = `(Batch, Height, Width, Channels)`
  - PyTorch format: `NCHW` = `(Batch, Channels, Height, Width)`
- **Outputs:**
  - Value: `(3,)` - Win/Draw/Loss probabilities (softmax)
  - Policy: `(4672,)` - Move probabilities (logits)

### Detailed Structure

```
1. CNN Stem:
   - Initial Conv: 34 → 128 channels (3×3, no bias)
   - BatchNorm: 128 channels
   - 4 Residual Blocks:
     * Block 0-1: 128 → 128 (no projection)
     * Block 2: 128 → 256 (with 1×1 projection)
     * Block 3: 256 → 256 (no projection)

2. Transformer Body (6 layers):
   - MultiHeadAttention: 8 heads, key_dim=32, embed_dim=256
   - Feed-Forward: 256 → 1024 → 256
   - LayerNorm (eps=1e-6) after attention and FFN

3. Value Head:
   - GlobalAveragePooling1D → Dense(256) → Dense(3) with softmax

4. Policy Head:
   - Reshape to (8, 8, 256) → Conv2D(256→2, 1×1) → Flatten → Dense(128→4672)
```

### Parameter Counts
- **PyTorch:** 8,137,797 parameters (trainable)
- **Keras:** 8,143,045 parameters total
  - Trainable: 8,139,717
  - Non-trainable (BatchNorm stats): 3,328
- **Difference:** 5,248 = 3,328 (stats) + 1,920 (redundant Conv biases)

---

## 2. Current Conversion Status

### ✅ What's Working

1. **CNN Layers:** Value outputs are very close (max diff ~0.001)
2. **Weight Loading:** 100% coverage (135/135 parameters loaded)
3. **Q/K/V Projections:** Mathematically verified correct (test shows 0.000000 difference)
4. **Input Layout:** Verified correct NCHW conversion
5. **Architecture Match:** Structures are identical

### ❌ The Problem

**Policy outputs differ significantly:**
- Max difference: **10-13** (very large)
- Mean difference: **0.8-1.0** (moderate)
- Top 10 moves: **0/10 overlap** (completely different predictions)
- Top move in Keras ranks **4302nd** in PyTorch (out of 4672)

**Value outputs are close:**
- Max difference: **~0.001** (excellent)
- This suggests CNN and most of transformer are correct

---

## 3. Detailed Output Comparison Data

### Starting Position

```
Value Outputs:
  Keras:   [0.4697, 0.4080, 0.1226]
  PyTorch: [0.4687, 0.4092, 0.1221]
  Max diff: 0.001238 ✓

Policy Outputs:
  Max difference:  12.374925
  Mean difference:  1.049754
  Median difference: 0.690881
  95th percentile: 3.042788
  99th percentile: 4.552093

Top 10 Moves:
  Keras:   [877, 804, 496, 731, 657, 1022, 950, 129, 876, 658]
  PyTorch: [1835, 1919, 2028, 1905, 2940, 3010, 3011, 2500, 1874, 2939]
  Overlap: 0/10 ✗

Top Move Analysis:
  Keras top move (877):
    Keras value:   6.199219
    PyTorch value: -5.154656
    Difference:    11.353875
    Rank in PyTorch: 4302nd (out of 4672)

  PyTorch top move (1835):
    Keras value:   -4.644531
    PyTorch value: 1.251661
    Difference:    5.896192
    Rank in Keras: 4363rd (out of 4672)

Difference Distribution:
  < 0.1:        78 (1.7%)
  0.1-1.0:    3482 (74.5%)
  1.0-5.0:    1076 (23.0%)
  >= 5.0:       36 (0.8%)

Scale Analysis:
  Keras policy range:   [-7.851562, 6.199219]
  PyTorch policy range: [-11.263704, 1.251661]
  Keras policy mean:    -3.770183
  PyTorch policy mean:  -3.295345
```

### After e4 Position

```
Policy Outputs:
  Max difference:  10.469721
  Mean difference:  0.878991
  Top 10 overlap: 0/10

Top Moves:
  Keras:   [3679, 3825, 3824, 3678, 3751, ...]
  PyTorch: [2531, 2532, 5, 2491, 1978, ...]
```

### Ruy Lopez Position

```
Policy Outputs:
  Max difference:  12.126089
  Mean difference:  0.767316
  Top 10 overlap: 0/10
```

---

## 4. Weight Conversion Details

### MHA Output Projection (FIXED per document)

**Keras Weight Format:**
```python
layers/multi_head_attention/output_dense/vars/0: shape (8, 32, 256)
# = (num_heads, key_dim, embed_dim)
```

**Current Conversion (per document):**
```python
# Step 1: Reshape to collapse head and key dimensions
w_reshaped = np.reshape(w_keras, (-1, w_keras.shape[-1]))  # (256, 256)

# Step 2: Transpose for PyTorch Linear layer
w_pytorch = np.transpose(w_reshaped)  # (256, 256)
```

**Verification:** This matches the document's exact code and has been verified equivalent.

### Q/K/V Projections (VERIFIED CORRECT)

**Keras Weight Format:**
```python
layers/multi_head_attention/query_dense/vars/0: shape (256, 8, 32)
layers/multi_head_attention/key_dense/vars/0: shape (256, 8, 32)
layers/multi_head_attention/value_dense/vars/0: shape (256, 8, 32)
```

**Conversion:**
```python
# For each (Q, K, V):
# 1. Reshape: (256, 8, 32) -> (256, 256)
q_r = q_kernel.reshape(256, 256)
# 2. Transpose: (256, 256) -> (256, 256) for PyTorch
q_t = q_r.T
# 3. Concatenate: [Q; K; V] -> (768, 256)
in_proj_weight = np.concatenate([q_t, k_t, v_t], axis=0)
```

**Mathematical Verification:** Tested with dummy input, shows 0.000000 difference ✓

### Q/K/V Biases

**Keras Format:**
```python
query_dense/vars/1: shape (8, 32)
key_dense/vars/1: shape (8, 32)
value_dense/vars/1: shape (8, 32)
```

**Conversion:**
```python
# Flatten each: (8, 32) -> (256,)
# Concatenate: [Q_bias; K_bias; V_bias] -> (768,)
in_proj_bias = np.concatenate([
    q_bias.flatten(),
    k_bias.flatten(),
    v_bias.flatten()
])
```

---

## 5. Files to Include in Research

### Core Model Files
1. **`src/transformer_model.py`** - Keras model definition
   - Lines 28-42: Transformer encoder block
   - Lines 45-99: Full model architecture
   - Key: Uses `tf.keras.layers.MultiHeadAttention` with `num_heads=8, key_dim=32`

2. **`src/transformer_model_pytorch.py`** - PyTorch model definition
   - Lines 81-140: TransformerEncoderBlock
   - Lines 143-289: HybridChessModel
   - Key: Uses `nn.MultiheadAttention` with `embed_dim=256, num_heads=8`

3. **`scripts/convert_keras_to_pytorch.py`** - Weight conversion script
   - Lines 175-242: MHA weight conversion (Q/K/V and output projection)
   - Current implementation matches document's code

### Test/Verification Files
4. **`scripts/compare_with_pytorch.py`** - Output comparison script
5. **`scripts/analyze_policy_differences.py`** - Detailed difference analysis
6. **`scripts/test_mha_conversion.py`** - Q/K/V conversion verification
7. **`scripts/verify_mha_hooks.py`** - Hook-based verification attempt

### Data Files
8. **`saved_models/keras_outputs.json`** - Pre-extracted Keras outputs for comparison
9. **`src/weights/best_model.keras`** - Original Keras model file
10. **`saved_models/best_model_pytorch.pt`** - Converted PyTorch model

### Configuration
11. **`src/move_mapping.py`** - Action space definition (4672 moves)
12. **`src/encoder.py`** - Board encoding (34 channels)

---

## 6. Key Code Snippets

### Keras MHA Usage
```python
# From src/transformer_model.py, line 31-33
attention_output = tf.keras.layers.MultiHeadAttention(
    num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
)(inputs, inputs)
```

### PyTorch MHA Usage
```python
# From src/transformer_model_pytorch.py, line 102-107
self.attention = nn.MultiheadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    dropout=dropout,
    batch_first=True
)
# Usage: attn_out, _ = self.attention(x, x, x, need_weights=False)
```

### Current MHA Output Projection Conversion
```python
# From scripts/convert_keras_to_pytorch.py, lines 227-236
if out_kernel is not None:
    # EXACT conversion per document: reshape then transpose
    out_reshaped = np.reshape(out_kernel, (-1, out_kernel.shape[-1]))
    out_proj_weight = np.transpose(out_reshaped)
    state_dict[f'{pt_prefix}.attention.out_proj.weight'] = torch.from_numpy(
        out_proj_weight.astype(np.float32).copy()
    )
```

### Q/K/V Conversion
```python
# From scripts/convert_keras_to_pytorch.py, lines 189-204
q_r = q_kernel.reshape(embed_dim, embed_dim)
k_r = k_kernel.reshape(embed_dim, embed_dim)
v_r = v_kernel.reshape(embed_dim, embed_dim)

in_proj_weight = np.concatenate([
    transpose_dense_weight(q_r),
    transpose_dense_weight(k_r),
    transpose_dense_weight(v_r)
], axis=0)
```

---

## 7. What We've Tried

### ✅ Applied Fixes (from document)
1. **MHA Output Projection:** Reshape then transpose (exact document code)
2. **Float32 Precision:** Ensured all weights are float32
3. **Input Layout:** Verified NCHW conversion
4. **LayerNorm Epsilon:** Verified matches (1e-6)

### ✅ Verified
1. **Q/K/V Conversion:** Mathematically tested, shows 0.000000 difference
2. **Parameter Counts:** Accounted for BatchNorm stats and biases
3. **Architecture Match:** Structures are identical
4. **Value Head:** Works correctly (max diff ~0.001)

### ❌ Still Failing
1. **Policy Outputs:** Completely different (0/10 top move overlap)
2. **Hook Verification:** Couldn't capture input to out_proj correctly (environment issue)

---

## 8. Research Questions

### Primary Questions

1. **How does Keras 3.x MultiHeadAttention actually compute attention?**
   - Does it concatenate or sum head outputs before the output projection?
   - What is the exact computation order?
   - Are there any scaling factors or normalization we're missing?

2. **How does PyTorch nn.MultiheadAttention compute attention?**
   - Exact computation order
   - How are head outputs combined?
   - Any differences in scaling or normalization?

3. **Q/K/V Weight Conversion - Are we missing something?**
   - We verified the mathematical conversion is correct
   - But are the weights being used in the right order?
   - Is there a head ordering issue?

4. **Attention Computation Differences:**
   - Scaling factor differences (1/sqrt(d_k))?
   - Masking differences?
   - Dropout application differences?

### Secondary Questions

5. **Input to Output Projection:**
   - What is the exact format of input to out_proj in both frameworks?
   - How should we verify they match?

6. **Bias Handling:**
   - Are biases being applied correctly?
   - Any differences in bias application order?

7. **Numerical Precision:**
   - Could float32 vs float64 cause accumulation errors?
   - Are there any operations that amplify small differences?

---

## 9. Intermediate Activation Statistics

### PyTorch Model (from hooks)

```
Initial Conv Output:
  Shape: (1, 128, 8, 8)
  Mean: -0.143068, Std: 0.766674

Initial BN Output:
  Shape: (1, 128, 8, 8)
  Mean: -0.325446, Std: 1.541125

Residual Block 0 Output:
  Shape: (1, 128, 8, 8)
  Mean: 0.101436, Std: 0.761163

Residual Block Last Output:
  Shape: (1, 256, 8, 8)
  Mean: 0.023215, Std: 0.438823

Transformer Layer 0 Output:
  Shape: (1, 64, 256)
  Mean: -0.013064, Std: 0.421109

Transformer Layer Last Output:
  Shape: (1, 64, 256)
  Mean: -0.028557, Std: 0.829760

Value FC1 Output:
  Shape: (1, 256)
  Mean: -1.383995, Std: 1.838441

Policy Conv Output:
  Shape: (1, 2, 8, 8)
  Mean: 0.566046, Std: 0.926396
```

### MHA Weight Statistics

```
Q/K/V Weights (in_proj_weight):
  Shape: (768, 256)
  Q portion (0:256) mean: -0.000549
  K portion (256:512) mean: 0.000236
  V portion (512:768) mean: 0.000245

Output Projection (out_proj_weight):
  Shape: (256, 256)
  Mean: -0.000027
  Std: 0.080850
  First value: -0.004109

Column Structure (head arrangement):
  Head 0 (cols 0:32): mean=0.001854
  Head 1 (cols 32:64): mean=-0.000420
  Head 2 (cols 64:96): mean=-0.000270
  Head 3 (cols 96:128): mean=-0.000340
  Head 4 (cols 128:160): mean=0.000968
  Head 5 (cols 160:192): mean=0.000171
  Head 6 (cols 192:224): mean=-0.001944
  Head 7 (cols 224:256): mean=-0.000238
```

---

## 10. Environment Details

### Keras Environment (`tf`)
- TensorFlow/Keras 3.x
- Model file: `src/weights/best_model.keras`
- Uses `swish` activation (mapped to `tf.nn.silu`)

### PyTorch Environment (`chess`)
- PyTorch 2.6.0+cu124
- Python 3.10
- Model file: `saved_models/best_model_pytorch.pt`

### Conversion Process
- Two-phase: Extract weights in `tf` env, load in `chess` env
- Direct H5 file reading to avoid Keras deserialization issues

---

## 11. Critical Observations

1. **Value Head Works:** Max diff ~0.001 suggests CNN and most transformer are correct
2. **Policy Head Fails:** Completely different predictions (0/10 overlap)
3. **Both Use Same Transformer Output:** The difference must be in how the transformer output is processed OR the transformer output itself is wrong
4. **Q/K/V Math Verified:** Test shows 0.000000 difference, but maybe the weights aren't being used correctly?
5. **Output Projection Matches Document:** We've applied the exact code from the research document

---

## 12. Hypothesis

**Most Likely:** The issue is in how PyTorch's `nn.MultiheadAttention` computes attention vs Keras 3.x's `MultiHeadAttention`. Even though Q/K/V weights convert correctly mathematically, the attention computation itself might:
- Use different scaling factors
- Combine heads differently
- Apply normalization differently
- Have different bias application

**Alternative:** There's a subtle issue in Q/K/V weight ordering or head arrangement that our tests didn't catch.

---

## 13. Next Steps for Research

1. **Compare Keras 3.x MHA source code with PyTorch MHA source code**
   - Exact computation formulas
   - Scaling factors
   - Head combination method

2. **Extract intermediate activations from both models**
   - Input to attention
   - Q/K/V projections
   - Attention scores
   - Attention output (before out_proj)
   - Input to out_proj

3. **Verify head ordering**
   - Are heads in the same order in both frameworks?
   - Is the concatenation order correct?

4. **Check for implementation differences**
   - Dropout application
   - Normalization placement
   - Bias handling

---

## 14. Test Commands

```bash
# Run comparison
conda activate chess
python scripts/compare_with_pytorch.py

# Detailed analysis
python scripts/analyze_policy_differences.py

# Verify conversion
python scripts/convert_keras_to_pytorch.py --keras-model src/weights/best_model.keras --output saved_models/best_model_pytorch.pt

# Test MHA conversion
python scripts/test_mha_conversion.py
```

---

## 15. Key Metrics Summary

| Metric | Value | Status |
|--------|-------|--------|
| Weight Coverage | 100% (135/135) | ✅ |
| Value Max Diff | ~0.001 | ✅ Excellent |
| Policy Max Diff | 10-13 | ❌ Very Large |
| Policy Mean Diff | 0.8-1.0 | ❌ Moderate |
| Top 10 Overlap | 0/10 | ❌ Complete Mismatch |
| Parameter Count Match | ✓ (accounting for stats) | ✅ |

---

**END OF RESEARCH CONTEXT**

