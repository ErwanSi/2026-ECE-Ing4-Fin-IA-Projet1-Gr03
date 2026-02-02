from src.strategies.vwap import vwap_schedule
print("\n--------------------------------------------------------------------------")
print("Running VWAP test...")

Q = 60
volumes = [10, 40, 200, 100, 100]  # exemple marché

print("Volumes:", volumes)
print(vwap_schedule(Q=Q, volumes=volumes))

print("\nWith participation_rate=0.2 (cap each slice to 20% of market volume)")
print("--------------------------------------------------------------------------\n")
