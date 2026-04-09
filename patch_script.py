with open("ev_dashboard.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "np.random.randint(1, 11, n_test)" in line:
        continue
    if "+ 0.5 * X_test[:, 4]" in line:
        continue
    new_lines.append(line)

with open("ev_dashboard.py", "w") as f:
    f.writelines(new_lines)
