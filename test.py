import subprocess

# List of x values
x_values = [10, 100, 500, 1000]

for x in x_values:
    n = x + 50
    command = f"python pacman.py -p ApproximateSARSAAgent -n {n} -x {x}"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)