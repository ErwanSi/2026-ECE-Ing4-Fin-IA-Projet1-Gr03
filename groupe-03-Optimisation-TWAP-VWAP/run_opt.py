from src.strategies.constrained_opt_cp import constrained_opt_cp_schedule

Q = 200
volumes = [10, 40, 200, 100, 100]

print("\n--------------------------------------------------------------------------")


print("\nRunning... CP-SAT OPT ")
print(constrained_opt_cp_schedule(
    Q=Q,
    volumes=volumes,
    participation_rate=0.7,
    w_impact=1,
    w_track=1
))




