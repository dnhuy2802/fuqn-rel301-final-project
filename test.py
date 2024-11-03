import subprocess

# List of x values
x_values = [10] # 50, 100

for x in x_values:
    n = x + 50
    # with out graphics
    command = f"python pacman.py -p ApproximateSARSAAgent -n {n} -x {x} -q"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)