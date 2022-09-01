import pickle
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib.pyplot as plt

def plot_sobols_first_second(filename):
    q = filename.split("processed_results_")[1].split(".pickle")[0]
    plotname = q+"_sobols.png"

    R = {}

    try:
        with open(filename, "rb") as f:
            R["results"] = pickle.load(f)
    
    except FileNotFoundError:
        print("Could not find file: ", filename)

    # Extract results
    ky = R["results"]["ky"]
    sobols_first_omega = R["results"]["sobols_first_omega"]
    sobols_first_gamma = R["results"]["sobols_first_gamma"]
    sobols_second_omega = R["results"]["sobols_second_omega"]
    sobols_second_gamma = R["results"]["sobols_second_gamma"]
    sobols_total_omega = R["results"]["sobols_total_omega"]
    sobols_total_gamma = R["results"]["sobols_total_gamma"]

    # LaTeX for plot legends
    symbols = {
        "fprim": r"$\kappa_n$",
        "tprim": r"$\kappa_t$",
        "pk": r"$p_k$",
        "shat": r"$\hat{s}$"
    }

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 6))
    #fig.suptitle("First order Sobol indices")

    for k in sobols_first_omega.keys():
        param = k.split("::")[1] # GS2 label
        symbol = symbols[param]
        axs[0, 0].plot(ky, sobols_first_omega[k], "o-", label=symbol)
        axs[0, 1].plot(ky, sobols_first_gamma[k], "o-", label=symbol)

    plotted_pairs = []

    for k1 in sobols_second_omega.keys():
        for k2 in sobols_second_omega[k1].keys():
            p1, p2 = k1.split("::")[1], k2.split("::")[1]
            if ((p1, p2) in plotted_pairs) or ((p2, p1) in plotted_pairs): 
                continue
            else:
                s1, s2 = symbols[p1], symbols[p2]
                plotted_pairs.append((p1, p2))
                axs[1, 0].plot(ky, sobols_second_omega[k1][k2], "o-", label=s1+", "+s2)
                axs[1, 1].plot(ky, sobols_second_gamma[k1][k2], "o-", label=s1+", "+s2)

    axs[0, 0].set_ylabel("First order Sobol indices")
    axs[1, 0].set_ylabel("Second order Sobol indices")
    axs[0, 0].set_title(r"$\omega_r/4$")
    axs[0, 1].set_title(r"$\gamma$")
    fig.supxlabel(r"$k_y\rho$")

    axs[0, 0].annotate("(a)", xy=(0.05, 0.85))
    axs[0, 1].annotate("(b)", xy=(0.05, 0.85))
    axs[1, 0].annotate("(c)", xy=(0.05, 0.85))
    axs[1, 1].annotate("(d)", xy=(0.05, 0.85))

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()

    plt.savefig(plotname, dpi=300)
    plt.show()

def plot_all_sobols(filename):
    q = filename.split("processed_results_")[1].split(".pickle")[0]
    plotname = q+"_sobols.png"

    R = {}

    try:
        with open(filename, "rb") as f:
            R["results"] = pickle.load(f)
    
    except FileNotFoundError:
        print("Could not find file: ", filename)

    # Extract results
    ky = R["results"]["ky"]
    sobols_first_omega = R["results"]["sobols_first_omega"]
    sobols_first_gamma = R["results"]["sobols_first_gamma"]
    sobols_second_omega = R["results"]["sobols_second_omega"]
    sobols_second_gamma = R["results"]["sobols_second_gamma"]
    sobols_total_omega = R["results"]["sobols_total_omega"]
    sobols_total_gamma = R["results"]["sobols_total_gamma"]

    # LaTeX for plot legends
    symbols = {
        "fprim": r"$\kappa_n$",
        "tprim": r"$\kappa_t$",
        "pk": r"$p_k$",
        "shat": r"$\hat{s}$"
    }

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(8, 8.5))
    #fig.suptitle("First order Sobol indices")

    for k in sobols_first_omega.keys():
        param = k.split("::")[1] # GS2 label
        symbol = symbols[param]
        axs[0, 0].plot(ky, sobols_first_omega[k], "o-", label=symbol)
        axs[0, 1].plot(ky, sobols_first_gamma[k], "o-", label=symbol)

    plotted_pairs = []

    for k1 in sobols_second_omega.keys():
        for k2 in sobols_second_omega[k1].keys():
            p1, p2 = k1.split("::")[1], k2.split("::")[1]
            if ((p1, p2) in plotted_pairs) or ((p2, p1) in plotted_pairs): 
                continue
            else:
                s1, s2 = symbols[p1], symbols[p2]
                plotted_pairs.append((p1, p2))
                axs[1, 0].plot(ky, sobols_second_omega[k1][k2], "o-", label=s1+", "+s2)
                axs[1, 1].plot(ky, sobols_second_gamma[k1][k2], "o-", label=s1+", "+s2)

    for k in sobols_total_omega.keys():
        param = k.split("::")[1] # GS2 label
        symbol = symbols[param]
        axs[2, 0].plot(ky, sobols_total_omega[k], "o-", label=symbol)
        axs[2, 1].plot(ky, sobols_total_gamma[k], "o-", label=symbol)

    axs[0, 0].set_ylabel("First order Sobol indices")
    axs[1, 0].set_ylabel("Second order Sobol indices")
    axs[2, 0].set_ylabel("Total Sobol indices")
    axs[0, 0].set_title(r"$\omega_r/4$")
    axs[0, 1].set_title(r"$\gamma$")
    fig.supxlabel(r"$k_y\rho$")

    axs[0, 0].annotate("(a)", xy=(0.20, 0.85))
    axs[0, 1].annotate("(b)", xy=(0.05, 0.85))
    axs[1, 0].annotate("(c)", xy=(0.05, 0.85))
    axs[1, 1].annotate("(d)", xy=(0.05, 0.85))
    axs[2, 0].annotate("(e)", xy=(0.20, 0.85))
    axs[2, 1].annotate("(f)", xy=(0.05, 0.85))

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    axs[2, 0].legend()
    axs[2, 1].legend()

    plt.savefig(plotname, dpi=300)
    plt.show()

def plot_means(filename):
    q = filename.split("processed_results_")[1].split(".pickle")[0]
    plotname = q+"_means.png"

    R = {}

    try:
        with open(filename, "rb") as f:
            R["results"] = pickle.load(f)
    
    except FileNotFoundError:
        print("Could not find file: ", filename)

    # Extract results
    ky = R["results"]["ky"]
    omega = R["results"]["omega"]
    omega_std = R["results"]["omega_std"]
    gamma = R["results"]["gamma"]
    gamma_std = R["results"]["gamma_std"]

    plt.figure()
    plt.plot(ky, omega, "o-", color="orange", label=r"$\omega_r/4$")
    plt.plot(ky, omega - omega_std, "--", color="orange")
    plt.plot(ky, omega + omega_std, "--", color="orange")
    plt.fill_between(ky, omega - omega_std, omega + omega_std, color="orange", alpha=0.4)
    plt.plot(ky, gamma, "o-", color="blue", label=r"$\gamma$")
    plt.plot(ky, gamma - gamma_std, "--", color="blue")
    plt.plot(ky, gamma + gamma_std, "--", color="blue")
    plt.fill_between(ky, gamma - gamma_std, gamma + gamma_std, color="blue", alpha=0.2)
    plt.legend()
    plt.xlabel(r"$k_y\rho$")
    plt.ylabel(r"$\gamma$" + " " + r"$[v_\mathrm{th}/L_{ne}]$")
    plt.savefig(plotname, dpi=300)
    plt.close()

if __name__ == "__main__":
    filenames = [
        "processed_results_fprim_tprim.pickle",
        "processed_results_pk_shat.pickle",
        "processed_results_fprim_tprim_pk_shat.pickle"
    ]

    for filename in filenames:
        # plot means for all three
        plot_means(filename)

        # plot total sobols for big sim only
        if "fprim_tprim_pk_shat" in filename:
            plot_all_sobols(filename)
        else:
            plot_sobols_first_second(filename)
        
