# TimeMixer: Concrete Explanation with Element-Level Examples

## Input: (2, 96, 3)
- **2 batches**
- **96 time steps** 
- **3 series**

---

## Step-by-Step Pipeline with Concrete Examples

### Step 1: Multi-Scale Downsampling

**Purpose:** Create multiple temporal scales to capture patterns at different resolutions.

**Operation:**
```
Input: (2, 96, 3)
├─ Scale 0 (original): (2, 96, 3) - Fine resolution, 96 time steps
└─ Scale 1 (downsampled): (2, 48, 3) - Coarse resolution, 48 time steps
```

**Concrete Example:**
```
Scale 0, batch 0, series 0, first 5 steps:
  [-0.44, -1.021, 0.437, 0.165, -0.648, ...]

Scale 1, batch 0, series 0, first 3 steps:
  [-0.73, 0.301, -0.072, ...]
  
Where: -0.73 = average(-0.44, -1.021)
       0.301 = average(0.437, 0.165)
```

**Method:** Average pooling with window size 2 (every 2 consecutive values → 1 value)

---

### Step 2: Normalization (RevIN)

**Purpose:** Normalize each batch independently to handle different scales.

**Operation:**
```
For each batch:
  mean = mean(x, dim=time)  # Compute mean across time dimension
  std = std(x, dim=time)
  x_norm = (x - mean) / std
```

**Concrete Example:**
```
Scale 0, batch 0:
  Original mean per series: [-0.115, 0.02, 0.088]
  Original std per series:  [0.835, 0.686, 0.725]
  
After normalization, each series has mean ≈ 0 and std ≈ 1
```

---

### Step 3: Embedding

**Purpose:** Project from input features (3) to model dimension (32).

**Operation:**
```
Linear(3 → 32): Each of the 3 series features → 32-dimensional representation
```

**Concrete Example:**
```
Before: (2, 96, 3) - 3 features per time step
After:  (2, 96, 32) - 32 features per time step

Each time step now has a richer representation in 32-dimensional space.
```

**Matrix Multiplication:**
- Weight matrix: `W_emb ∈ ℝ^(3 × 32)`
- For each time step: `x_embedded = x @ W_emb`
- Total: 96 time steps × 2 batches × 3 series = many matrix multiplications

---

### Step 4: Series Decomposition

**Purpose:** Split each scale into **seasonal** (short-term fluctuations) and **trend** (long-term direction) components.

**Operation:**
```
For each scale:
  trend = MovingAverage(x, window_size=25)  # Smooth the series
  season = x - trend                         # Residual (fluctuations)
```

**Concrete Example:**
```
Scale 0, batch 0, feature 0, first 5 steps:

  Original:  [1.538, 0.967, 0.241, 0.099, -0.02]
  Trend:     [0.29,  0.327, 0.357, 0.409,  0.47 ]  ← Smoothed (moving avg)
  Season:    [1.248, 0.639, -0.116, -0.31, -0.49]  ← Fluctuations
  
  Check: Season + Trend = Original ✓
```

**Intuition:**
- **Trend**: Captures slow-moving, long-term patterns (e.g., overall growth/decline)
- **Season**: Captures fast-moving, short-term patterns (e.g., daily/weekly cycles)

**Decomposition Method:**
- Moving average: Simple smoothing filter
- DFT-based: Uses frequency domain (top-k frequencies → season, rest → trend)

---

### Step 5: Multi-Scale Season Mixing (Bottom-Up)

**Purpose:** Mix seasonal patterns from **fine → coarse** scales. High-resolution seasonal details flow into low-resolution representations.

**Operation Flow:**
```
Scale 0 (fine, 96 steps) ──[Downsample]──> Scale 1 (coarse, 48 steps)
        ↓                                        ↓
   Season[0]                                 Season[1]
        ↓                                        ↓
  (2, 32, 96)                            (2, 32, 48)
        ↓                                        ↓
   Linear(96→48) ────────────────────────────────┐
        ↓                                        ↓
   (2, 32, 48) ──[Add]──> Season[1] + downsampled = Mixed[1]
```

**Concrete Example:**
```
Initial state:
  out_high (scale 0): shape (2, 32, 96) - fine seasonal patterns
  out_low (scale 1):  shape (2, 32, 48) - coarse seasonal patterns

Downsampling operation:
  1. Apply Linear(96 → 48) to scale 0:
     - Input: (2, 32, 96) 
     - Reshape to (2×32, 96) = (64, 96)
     - Matrix multiply: (64, 96) @ W_down ∈ ℝ^(96 × 48) = (64, 48)
     - Reshape back: (2, 32, 48)
     
  2. Add to scale 1:
     out_low_mixed = out_low + downsampled_high

Concrete values (batch=0, feature=0):
  Original out_low[0, 0, :5]:     [1.005, -0.508, -0.829, -0.378, 0.129]
  Downsampled high[0, 0, :5]:     [0.078, -0.121,  0.186,  0.393, 0.038]
  Mixed out_low[0, 0, :5]:        [1.083, -0.628, -0.644,  0.015, 0.167]
                                    ↑
                              Element-wise addition
```

**Matrix Multiplications:**
- Downsampling layer: `Linear(96 → 48)` → `Linear(48 → 48)`
- Applied per feature (32 features)
- Total: 2 batches × 32 features = 64 matrix multiplications per scale mixing

**Why Bottom-Up?**
- Seasonal patterns (short-term fluctuations) are better captured at fine scales
- Aggregating fine seasonal information into coarse scales preserves important high-frequency details

---

### Step 6: Multi-Scale Trend Mixing (Top-Down)

**Purpose:** Mix trend patterns from **coarse → fine** scales. Low-resolution trend information flows into high-resolution representations.

**Operation Flow:**
```
Scale 1 (coarse, 48 steps) ──[Upsample]──> Scale 0 (fine, 96 steps)
        ↓                                        ↓
   Trend[1]                                  Trend[0]
        ↓                                        ↓
  (2, 32, 48)                            (2, 32, 96)
        ↓                                        ↓
   Linear(48→96) ────────────────────────────────┐
        ↓                                        ↓
   (2, 32, 96) ──[Add]──> Trend[0] + upsampled = Mixed[0]
```

**Concrete Example:**
```
Initial state:
  out_low (scale 1):  shape (2, 32, 48) - coarse trend patterns
  out_high (scale 0): shape (2, 32, 96) - fine trend patterns

Upsampling operation:
  1. Apply Linear(48 → 96) to scale 1:
     - Input: (2, 32, 48)
     - Reshape to (2×32, 48) = (64, 48)
     - Matrix multiply: (64, 48) @ W_up ∈ ℝ^(48 × 96) = (64, 96)
     - Reshape back: (2, 32, 96)
     
  2. Add to scale 0:
     out_high_mixed = out_high + upsampled_low

Concrete values (batch=0, feature=0):
  Original out_high[0, 0, :10]:    [0.29, 0.327, 0.357, 0.409, 0.47, ...]
  Upsampled low[0, 0, :10]:        [0.057, -0.03, -0.181, 0.073, -0.009, ...]
  Mixed out_high[0, 0, :10]:       [0.346, 0.298, 0.176, 0.481, 0.461, ...]
                                     ↑
                               Element-wise addition
```

**Matrix Multiplications:**
- Upsampling layer: `Linear(48 → 96)` → `Linear(96 → 96)`
- Applied per feature (32 features)
- Total: 2 batches × 32 features = 64 matrix multiplications per scale mixing

**Why Top-Down?**
- Trend patterns (long-term direction) are better captured at coarse scales
- Injecting coarse trend information into fine scales provides global context

---

### Step 7: Reconstruct and Combine

**Purpose:** Combine mixed seasonal and trend components back together.

**Operation:**
```
For each scale:
  combined = mixed_season + mixed_trend  # Element-wise addition
```

**Concrete Example:**
```
Scale 0, batch 0, time 0, feature 0:
  Season:  1.248  (from Step 5 - original fine seasonal pattern)
  Trend:   0.346  (from Step 6 - coarse trend mixed into fine)
  Combined: 1.595 = 1.248 + 0.346
  
Scale 1, batch 0, time 0, feature 0:
  Season:  1.083  (from Step 5 - fine seasonal pattern mixed into coarse)
  Trend:   0.623  (from Step 6 - original coarse trend pattern)
  Combined: 1.707 = 1.083 + 0.623
```

---

### Step 8: Future Multi-Predictor Mixing (Prediction)

**Purpose:** Each scale independently predicts the future horizon `h`.

**Operation:**
```
For each scale i:
  1. Permute: (B, T, C) → (B, C, T)
  2. Predict: Linear(T → h)  # Map time steps to horizon
  3. Permute back: (B, C, h) → (B, h, C)
  4. Project: Linear(C → N)  # Map features to output series
```

**Concrete Example:**
```
Scale 0:
  Input: (2, 96, 32)
  → Permute: (2, 32, 96)
  → Predict: Linear(96 → 24) → (2, 32, 24)
  → Permute: (2, 24, 32)
  → Project: Linear(32 → 3) → (2, 24, 3)
  
Scale 1:
  Input: (2, 48, 32)
  → Permute: (2, 32, 48)
  → Predict: Linear(48 → 24) → (2, 32, 24)
  → Permute: (2, 24, 32)
  → Project: Linear(32 → 3) → (2, 24, 3)

Predictions:
  Scale 0 prediction[0, 0, :]: [-0.336, -0.151, 0.356]  # For horizon=0
  Scale 1 prediction[0, 0, :]: [-0.226, -0.015, -0.155] # For horizon=0
```

**Matrix Multiplications:**
- **Prediction layer** (per scale): `Linear(T → h)`
  - Scale 0: `(2, 32, 96) @ W_pred0 ∈ ℝ^(96 × 24) = (2, 32, 24)`
  - Scale 1: `(2, 32, 48) @ W_pred1 ∈ ℝ^(48 × 24) = (2, 32, 24)`
  
- **Projection layer**: `Linear(32 → 3)`
  - Applied to both scales: `(2, 24, 32) @ W_proj ∈ ℝ^(32 × 3) = (2, 24, 3)`

---

### Step 9: Ensemble Predictions

**Purpose:** Sum predictions from all scales to create final ensemble forecast.

**Operation:**
```
Stack predictions: (B, h, N, num_scales)
Sum across scales: sum(dim=-1) → (B, h, N)
```

**Concrete Example:**
```
Stacked shape: (2, 24, 3, 2)
  - Dimension 0: 2 batches
  - Dimension 1: 24 forecast steps
  - Dimension 2: 3 series
  - Dimension 3: 2 scales

For batch=0, horizon=0, series=0:
  Scale 0 prediction: -0.336
  Scale 1 prediction: -0.226
  Final ensemble:     -0.562 = -0.336 + (-0.226)
```

**Why Sum?**
- Different scales capture complementary information:
  - Fine scales: Short-term details, high-frequency patterns
  - Coarse scales: Long-term trends, low-frequency patterns
- Summing allows both to contribute to final prediction

---

## Summary: All Matrix Multiplications

### Embedding Phase
1. **Embedding**: `Linear(3 → 32)`
   - Applied: 2 batches × 96 time steps × 2 scales
   - Operation: `(B, T, 3) @ W_emb ∈ ℝ^(3 × 32) = (B, T, 32)`

### Decomposition Phase
2. **Moving Average** (for decomposition): Convolution-based, no learnable matrix

### Mixing Phase (repeated for each PDM block, typically 4 times)
3. **Season Downsampling**: `Linear(96 → 48) → Linear(48 → 48)`
   - Applied: 2 batches × 32 features
   - Operation: `(B×C, 96) @ W_down ∈ ℝ^(96 × 48) = (B×C, 48)`

4. **Trend Upsampling**: `Linear(48 → 96) → Linear(96 → 96)`
   - Applied: 2 batches × 32 features
   - Operation: `(B×C, 48) @ W_up ∈ ℝ^(48 × 96) = (B×C, 96)`

5. **Cross-layer** (if channel_independence=0): `Linear(32 → 32)`
   - Applied per season/trend component

### Prediction Phase
6. **Prediction Layers** (per scale):
   - Scale 0: `Linear(96 → 24)` → `(B, C, 96) @ W_pred0 ∈ ℝ^(96 × 24) = (B, C, 24)`
   - Scale 1: `Linear(48 → 24)` → `(B, C, 48) @ W_pred1 ∈ ℝ^(48 × 24) = (B, C, 24)`

7. **Projection**: `Linear(32 → 3)`
   - Applied: `(B, h, 32) @ W_proj ∈ ℝ^(32 × 3) = (B, h, 3)`

---

## Key Insights

1. **Multi-Scale Processing**: TimeMixer processes the same input at multiple resolutions simultaneously, capturing both fine details and coarse trends.

2. **Asymmetric Mixing**:
   - **Season**: Fine → Coarse (high-frequency details preserved)
   - **Trend**: Coarse → Fine (long-term context injected)

3. **Decomposition Enables Separation**: By splitting season and trend, the model can:
   - Mix seasonal patterns independently (preserve short-term fluctuations)
   - Mix trend patterns independently (preserve long-term direction)

4. **Ensemble Prediction**: Multiple scales predict independently, then sum together - each scale contributes complementary forecasting capabilities.

5. **Fully MLP-Based**: All operations use simple linear layers and element-wise operations, making it efficient and easy to optimize.

---

## Final Output

**Input:** `(2, 96, 3)` - 2 batches, 96 time steps, 3 series  
**Output:** `(2, 24, 3)` - 2 batches, 24 forecast steps, 3 series

Each output value is an ensemble of predictions from multiple scales, where:
- Fine scales contribute short-term, detailed forecasts
- Coarse scales contribute long-term, trend-based forecasts
