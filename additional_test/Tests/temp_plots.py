import matplotlib.pyplot as plt


change_variable_at = [0, 30, 60, 120, 200]
change_variable = [1.0, 1.2, 1.4, 1.6, 1.6]

change_var  = [1.0]

j = 0
for i in range(1, len(change_variable_at)):
    while j < change_variable_at[i]:
        change_var.append(change_variable[i-1])
        j += 1

change_var = [1.0 for i in range(90)]
for i in range(1, 90):

    change_var[i] = change_var[i-1] +  65e-7


plt.plot(change_var)
plt.savefig("setting_0")