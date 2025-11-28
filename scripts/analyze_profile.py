# scripts/analyze_profile.py
import pstats

# --- CONFIGURATION ---
PROFILE_FILE = "worker_0_profile.prof"
TOP_N_FUNCTIONS = 20
# ---

print(f"--- Analyzing Profile Data from: {PROFILE_FILE} ---")

stats = pstats.Stats(PROFILE_FILE)

# Sort by 'cumulative time' to see which high-level functions take the most time overall
print(f"\n--- Top {TOP_N_FUNCTIONS} by Cumulative Time (incl. sub-functions) ---")
stats.sort_stats("cumulative").print_stats(TOP_N_FUNCTIONS)

# Sort by 'tottime' (total time) to see which specific functions are the slowest
print(f"\n--- Top {TOP_N_FUNCTIONS} by Total Time (exclusive, self-time) ---")
stats.sort_stats("tottime").print_stats(TOP_N_FUNCTIONS)