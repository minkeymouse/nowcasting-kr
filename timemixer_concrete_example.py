"""
Concrete Example: TimeMixer with Input Shape (2, 96, 3)
========================================================

This script demonstrates the TimeMixer pipeline with actual tensor operations
using concrete examples for 2 batches, 96 time steps, and 3 series.
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# SETUP: Concrete Input Example
# ============================================================================

print("=" * 80)
print("TIMEMIXER CONCRETE EXAMPLE")
print("=" * 80)
print("\nInput Configuration:")
print("  Batch size (B) = 2")
print("  Time steps (T) = 96")  
print("  Number of series (N) = 3")
print("  Input shape: (2, 96, 3)")

# Simulated input data
np.random.seed(42)
x_enc = torch.randn(2, 96, 3)  # [B, T, N]
print(f"\nInput tensor shape: {x_enc.shape}")

# Example: First few values of first batch, first series
print(f"\nFirst batch, first series (first 10 time steps):")
print(x_enc[0, :10, 0].numpy().round(3))

# ============================================================================
# STEP 1: Multi-Scale Downsampling
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: MULTI-SCALE DOWNSAMPLING")
print("=" * 80)

down_sampling_window = 2
down_sampling_layers = 1
down_sampling_method = "avg"

print(f"\nParameters:")
print(f"  down_sampling_window = {down_sampling_window}")
print(f"  down_sampling_layers = {down_sampling_layers}")
print(f"  down_sampling_method = '{down_sampling_method}'")

# Original scale (scale 0)
x_scale0 = x_enc  # Shape: (2, 96, 3)
print(f"\nScale 0 (original):")
print(f"  Shape: {x_scale0.shape}")
print(f"  Example values [batch=0, time=0:5, series=0]:")
print(f"    {x_scale0[0, 0:5, 0].numpy().round(3)}")

# Downsample using average pooling
# Convert to (B, C, T) for pooling
x_scale0_pool = x_scale0.permute(0, 2, 1)  # (2, 3, 96)
avg_pool = nn.AvgPool1d(kernel_size=down_sampling_window, stride=down_sampling_window)
x_scale1_pool = avg_pool(x_scale0_pool)  # (2, 3, 48)
x_scale1 = x_scale1_pool.permute(0, 2, 1)  # (2, 48, 3)

print(f"\nScale 1 (downsampled by 2):")
print(f"  Shape: {x_scale1.shape}")
print(f"  Example values [batch=0, time=0:3, series=0]:")
print(f"    {x_scale1[0, 0:3, 0].numpy().round(3)}")
print(f"\n  Note: Each value in scale 1 is the average of 2 consecutive values in scale 0")
print(f"  Example: scale1[0, 0, 0] = avg(scale0[0, 0:2, 0]) = {x_scale0[0, 0:2, 0].mean().item():.3f}")

# Store scales in list
x_enc_list = [x_scale0, x_scale1]
print(f"\nNumber of scales: {len(x_enc_list)}")

# ============================================================================
# STEP 2: Normalization (RevIN)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: NORMALIZATION (RevIN)")
print("=" * 80)

class SimpleRevIN:
    """Simplified RevIN for demonstration"""
    def __init__(self, num_features):
        self.num_features = num_features
        self.eps = 1e-5
    
    def forward(self, x, mode="norm"):
        # x shape: (B, T, N)
        if mode == "norm":
            # Compute mean and std across time dimension
            self.mean = x.mean(dim=1, keepdim=True)  # (B, 1, N)
            self.std = x.std(dim=1, keepdim=True) + self.eps  # (B, 1, N)
            x_norm = (x - self.mean) / self.std
            return x_norm
        elif mode == "denorm":
            return x * self.std + self.mean

norm_layer = SimpleRevIN(num_features=3)

# Normalize each scale
x_scale0_norm = norm_layer.forward(x_scale0, mode="norm")
x_scale1_norm = norm_layer.forward(x_scale1, mode="norm")

print(f"\nAfter normalization:")
print(f"  Scale 0: {x_scale0_norm.shape}")
print(f"  Scale 1: {x_scale1_norm.shape}")
print(f"\n  Normalization statistics for scale 0, batch 0:")
print(f"    Mean per series: {norm_layer.mean[0, 0, :].numpy().round(3)}")
print(f"    Std per series:  {norm_layer.std[0, 0, :].numpy().round(3)}")

# ============================================================================
# STEP 3: Embedding
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: EMBEDDING")
print("=" * 80)

d_model = 32
print(f"\nEmbedding dimension d_model = {d_model}")

# Simplified embedding: linear projection
# In reality, TimeMixer uses TokenEmbedding + optional TemporalEmbedding
embedding = nn.Linear(3, d_model)  # From N=3 features to d_model=32

x_scale0_emb = embedding(x_scale0_norm)  # (2, 96, 32)
x_scale1_emb = embedding(x_scale1_norm)  # (2, 48, 32)

print(f"\nAfter embedding:")
print(f"  Scale 0: {x_scale0_emb.shape}")
print(f"  Scale 1: {x_scale1_emb.shape}")
print(f"\n  Each time step now has {d_model} features instead of 3")

enc_out_list = [x_scale0_emb, x_scale1_emb]

# ============================================================================
# STEP 4: DECOMPOSITION (Trend-Seasonal Split)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: SERIES DECOMPOSITION")
print("=" * 80)

moving_avg_kernel = 25  # Window size for moving average
print(f"\nMoving average kernel size = {moving_avg_kernel}")

class SimpleMovingAvg:
    """Simplified moving average for demonstration"""
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
    
    def forward(self, x):
        # x shape: (B, T, C)
        # Apply 1D convolution for moving average
        # Use padding to maintain length
        padding = self.kernel_size // 2
        weights = torch.ones(1, 1, self.kernel_size) / self.kernel_size
        x_conv = x.permute(0, 2, 1)  # (B, C, T)
        # Simple implementation: for each position, take mean of window
        trend = torch.zeros_like(x)
        for t in range(x.shape[1]):
            start = max(0, t - padding)
            end = min(x.shape[1], t + padding + 1)
            trend[:, t, :] = x[:, start:end, :].mean(dim=1)
        return trend

# For demonstration, use a simpler approach
def simple_decompose(x, kernel_size=25):
    """Decompose x into trend (moving avg) and season (residual)"""
    # x shape: (B, T, C)
    trend = torch.zeros_like(x)
    for t in range(x.shape[1]):
        start = max(0, t - kernel_size // 2)
        end = min(x.shape[1], t + kernel_size // 2 + 1)
        trend[:, t, :] = x[:, start:end, :].mean(dim=1)
    season = x - trend  # Residual = original - trend
    return season, trend

# Decompose each scale
print("\nDecomposing each scale into seasonal and trend components...")

season_list = []
trend_list = []

for i, enc_out in enumerate(enc_out_list):
    season, trend = simple_decompose(enc_out, kernel_size=moving_avg_kernel)
    season_list.append(season)
    trend_list.append(trend)
    print(f"\n  Scale {i} (shape {enc_out.shape}):")
    print(f"    Season (residual): {season.shape}")
    print(f"    Trend (smoothed):  {trend.shape}")
    print(f"    Example - Scale {i}, batch 0, feature 0, first 5 time steps:")
    print(f"      Original:  {enc_out[0, :5, 0].detach().numpy().round(3)}")
    print(f"      Trend:     {trend[0, :5, 0].detach().numpy().round(3)}")
    print(f"      Season:    {season[0, :5, 0].detach().numpy().round(3)}")
    print(f"      Check (season + trend): {(season[0, :5, 0] + trend[0, :5, 0]).detach().numpy().round(3)}")

# Permute for mixing: (B, T, C) -> (B, C, T) for linear layers
season_list_transposed = [s.permute(0, 2, 1) for s in season_list]  # (B, C, T)
trend_list_transposed = [t.permute(0, 2, 1) for t in trend_list]    # (B, C, T)

print(f"\nAfter permutation for mixing:")
print(f"  Season scales: {[s.shape for s in season_list_transposed]}")
print(f"  Trend scales:  {[t.shape for t in trend_list_transposed]}")

# ============================================================================
# STEP 5: MULTI-SCALE SEASON MIXING (Bottom-Up)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MULTI-SCALE SEASON MIXING (Bottom-Up: Fine → Coarse)")
print("=" * 80)

print("\nSeason mixing flow:")
print("  Scale 0 (fine, 96 steps) → Scale 1 (coarse, 48 steps)")
print("  Information flows from high-resolution to low-resolution")

# Create downsampling layer: (96, 48) linear layer
seq_len = 96
down_sampling_window = 2
season_down_layer = nn.Sequential(
    nn.Linear(96, 48),   # Downsample time dimension: 96 → 48
    nn.GELU(),
    nn.Linear(48, 48)
)

print(f"\nDownsampling layer structure:")
print(f"  Linear(96 → 48) → GELU → Linear(48 → 48)")

# Mixing process
out_season_list = []

# Start with scale 0 (fine)
out_high = season_list_transposed[0]  # (2, 32, 96) - batch, features, time
out_low = season_list_transposed[1]   # (2, 32, 48)

print(f"\nInitial state:")
print(f"  out_high (scale 0): {out_high.shape}")
print(f"  out_low (scale 1):  {out_low.shape}")

# Apply downsampling to out_high and add to out_low
print(f"\nMixing operation:")
print(f"  1. Apply downsampling to scale 0: out_high (B, C, 96) → (B, C, 48)")
# Linear layer operates on last dimension, so we need to reshape
# out_high: (B, C, T) -> (B*C, T) -> Linear(T->T') -> (B*C, T') -> (B, C, T')
B, C, T = out_high.shape
out_high_reshaped = out_high.permute(0, 1, 2).contiguous().view(B * C, T)  # (B*C, 96)
downsampled_high_flat = season_down_layer(out_high_reshaped)  # (B*C, 48)
downsampled_high = downsampled_high_flat.view(B, C, 48)  # (B, C, 48)
print(f"     Result shape: {downsampled_high.shape}")

print(f"  2. Add to scale 1: out_low = out_low + downsampled_high")
out_low_mixed = out_low + downsampled_high
print(f"     Result shape: {out_low_mixed.shape}")

print(f"\n  Concrete example (batch=0, feature=0):")
print(f"    Original out_low[0, 0, :5]:     {out_low[0, 0, :5].detach().numpy().round(3)}")
print(f"    Downsampled high[0, 0, :5]:     {downsampled_high[0, 0, :5].detach().numpy().round(3)}")
print(f"    Mixed out_low[0, 0, :5]:        {out_low_mixed[0, 0, :5].detach().numpy().round(3)}")

out_season_list = [out_high, out_low_mixed]

print(f"\nFinal season outputs:")
for i, s in enumerate(out_season_list):
    print(f"  Scale {i}: {s.shape}")

# ============================================================================
# STEP 6: MULTI-SCALE TREND MIXING (Top-Down)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: MULTI-SCALE TREND MIXING (Top-Down: Coarse → Fine)")
print("=" * 80)

print("\nTrend mixing flow:")
print("  Scale 1 (coarse, 48 steps) → Scale 0 (fine, 96 steps)")
print("  Information flows from low-resolution to high-resolution")

# Create upsampling layer: (48, 96) linear layer
trend_up_layer = nn.Sequential(
    nn.Linear(48, 96),   # Upsample time dimension: 48 → 96
    nn.GELU(),
    nn.Linear(96, 96)
)

print(f"\nUpsampling layer structure:")
print(f"  Linear(48 → 96) → GELU → Linear(96 → 96)")

# Mixing process (reverse order)
trend_list_reverse = trend_list_transposed.copy()
trend_list_reverse.reverse()  # Start from coarsest scale

out_trend_list = []

# Start with scale 1 (coarse)
out_low = trend_list_reverse[0]   # (2, 32, 48) - the coarse scale
out_high = trend_list_reverse[1]  # (2, 32, 96) - the fine scale

print(f"\nInitial state:")
print(f"  out_low (scale 1, coarse):  {out_low.shape}")
print(f"  out_high (scale 0, fine):   {out_high.shape}")

# Apply upsampling to out_low and add to out_high
print(f"\nMixing operation:")
print(f"  1. Apply upsampling to scale 1: out_low (B, C, 48) → (B, C, 96)")
# Linear layer operates on last dimension
B, C, T_low = out_low.shape
out_low_reshaped = out_low.permute(0, 1, 2).contiguous().view(B * C, T_low)  # (B*C, 48)
upsampled_low_flat = trend_up_layer(out_low_reshaped)  # (B*C, 96)
upsampled_low = upsampled_low_flat.view(B, C, 96)  # (B, C, 96)
print(f"     Result shape: {upsampled_low.shape}")

print(f"  2. Add to scale 0: out_high = out_high + upsampled_low")
out_high_mixed = out_high + upsampled_low
print(f"     Result shape: {out_high_mixed.shape}")

print(f"\n  Concrete example (batch=0, feature=0):")
print(f"    Original out_high[0, 0, :10]:    {out_high[0, 0, :10].detach().numpy().round(3)}")
print(f"    Upsampled low[0, 0, :10]:        {upsampled_low[0, 0, :10].detach().numpy().round(3)}")
print(f"    Mixed out_high[0, 0, :10]:       {out_high_mixed[0, 0, :10].detach().numpy().round(3)}")

out_trend_list = [out_high_mixed, out_low]

print(f"\nFinal trend outputs:")
for i, t in enumerate(out_trend_list):
    print(f"  Scale {i}: {t.shape}")

# ============================================================================
# STEP 7: RECONSTRUCT AND COMBINE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: RECONSTRUCT AND COMBINE (Season + Trend)")
print("=" * 80)

# Permute back: (B, C, T) -> (B, T, C)
out_season_list_final = [s.permute(0, 2, 1) for s in out_season_list]
out_trend_list_final = [t.permute(0, 2, 1) for t in out_trend_list]

# Combine season and trend
mixed_outputs = []
for season, trend in zip(out_season_list_final, out_trend_list_final):
    combined = season + trend  # Element-wise addition
    mixed_outputs.append(combined)

print(f"\nCombined outputs (season + trend):")
for i, out in enumerate(mixed_outputs):
    print(f"  Scale {i}: {out.shape}")
    print(f"    Example [batch=0, time=0, feature=0]:")
    print(f"      Season: {out_season_list_final[i][0, 0, 0].item():.3f}")
    print(f"      Trend:  {out_trend_list_final[i][0, 0, 0].item():.3f}")
    print(f"      Combined: {out[0, 0, 0].item():.3f}")

# ============================================================================
# STEP 8: FUTURE MULTI-PREDICTOR MIXING (Prediction)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: FUTURE MULTI-PREDICTOR MIXING (Prediction)")
print("=" * 80)

h = 24  # Forecast horizon
print(f"\nForecast horizon h = {h}")

# Create prediction layers for each scale
predict_layers = nn.ModuleList([
    nn.Linear(96, h),  # Scale 0: 96 time steps → h predictions
    nn.Linear(48, h),  # Scale 1: 48 time steps → h predictions
])

# Project to output dimension
projection_layer = nn.Linear(32, 3)  # d_model → n_series

dec_out_list = []
for i, (enc_out, mixed_out) in enumerate(zip(enc_out_list, mixed_outputs)):
    print(f"\nScale {i} prediction:")
    print(f"  Input shape: {mixed_out.shape}")
    
    # Permute: (B, T, C) -> (B, C, T) for linear layer
    mixed_out_transposed = mixed_out.permute(0, 2, 1)  # (B, C, T)
    print(f"  After permute: {mixed_out_transposed.shape}")
    
    # Predict: time dimension T → h
    pred = predict_layers[i](mixed_out_transposed)  # (B, C, h)
    print(f"  After prediction layer: {pred.shape}")
    
    # Permute back: (B, C, h) -> (B, h, C)
    pred = pred.permute(0, 2, 1)  # (B, h, C)
    print(f"  After permute back: {pred.shape}")
    
    # Project to n_series
    pred_final = projection_layer(pred)  # (B, h, 3)
    print(f"  After projection: {pred_final.shape}")
    
    dec_out_list.append(pred_final)

print(f"\nPredictions from each scale:")
for i, pred in enumerate(dec_out_list):
    print(f"  Scale {i}: {pred.shape}")
    print(f"    Example [batch=0, horizon=0, series]: {pred[0, 0, :].detach().numpy().round(3)}")

# ============================================================================
# STEP 9: ENSEMBLE PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: ENSEMBLE PREDICTIONS (Sum Across Scales)")
print("=" * 80)

# Stack predictions: (B, h, N, num_scales)
pred_stack = torch.stack(dec_out_list, dim=-1)  # (2, 24, 3, 2)
print(f"\nStacked predictions shape: {pred_stack.shape}")
print(f"  Dimension breakdown: (batch={pred_stack.shape[0]}, horizon={pred_stack.shape[1]}, series={pred_stack.shape[2]}, scales={pred_stack.shape[3]})")

# Sum across scales (last dimension)
final_pred = pred_stack.sum(dim=-1)  # (2, 24, 3)
print(f"\nFinal ensemble prediction shape: {final_pred.shape}")

print(f"\nExample ensemble calculation (batch=0, horizon=0, series=0):")
print(f"  Scale 0 prediction: {dec_out_list[0][0, 0, 0].item():.3f}")
print(f"  Scale 1 prediction: {dec_out_list[1][0, 0, 0].item():.3f}")
print(f"  Ensemble (sum):     {final_pred[0, 0, 0].item():.3f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Input:  (2, 96, 3) - 2 batches, 96 time steps, 3 series
Output: (2, 24, 3) - 2 batches, 24 forecast steps, 3 series

Key Operations:
1. Multi-scale downsampling: Created 2 scales (96 → 48 time steps)
2. Decomposition: Each scale split into season (residual) and trend (smoothed)
3. Season mixing: Fine → Coarse (scale 0 → scale 1) via downsampling linear layers
4. Trend mixing: Coarse → Fine (scale 1 → scale 0) via upsampling linear layers
5. Reconstruction: Season + Trend per scale
6. Prediction: Each scale predicts h=24 steps independently
7. Ensemble: Sum predictions from all scales

Matrix Multiplications:
- Embedding: Linear(N=3 → d_model=32)
- Season downsampling: Linear(T=96 → T=48)
- Trend upsampling: Linear(T=48 → T=96)
- Prediction: Linear(T → h=24) per scale
- Projection: Linear(d_model=32 → N=3)
""")
