workers = 100
maxWorkers = 100
value = 100
#new_effort = value * (1 + (1.0 / maxWorkers) * (workers - 1))
new_effort = value * (1 + (1 / maxWorkers)) ** (workers - 1) # Grans law

duration = new_effort / workers

print(new_effort)
print(duration)
print(duration * 2)