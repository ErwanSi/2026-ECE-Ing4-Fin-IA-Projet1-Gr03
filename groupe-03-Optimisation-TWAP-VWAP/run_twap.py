from src.strategies.twap import twap_schedule
print("--------------------------------------------------------------------------")

print("Running TWAP test...")
print(twap_schedule(Q=250, N=10, max_per_slice=None))

