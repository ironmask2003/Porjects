# Funzione che controlla riga per riga se due file sono uguali
def check_out(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        for line1, line2 in zip(f1, f2):
            if line1 != line2:
                return False
    return True


# Prende in input un file .sh e lo esegue con un input per esempio, deve eseguire ./ciao.sh 100D
def exec_file(file, test):
    import subprocess

    subprocess.run(["sh", file, test])
    return


if __name__ == "__main__":
    # Prende in input una stringa
    vers = input("Inserisci la versione del file da controllare: ")
    test = input("Inserisci la versione del file di test: ")

    # Esecuzione della versione sequenziale
    seq_file = "seq_run.sh"
    exec_file(seq_file, test)

    # Esecuzione della versione scelta
    sh_file = f"{vers}_run.sh"
    exec_file(sh_file, test)

    if vers == "omp":
        print(
            check_out(
                f"output_files/seq/output{test}.txt",
                f"output_files/omp/output{test}.txt",
            )
        )
    elif vers == "mpi":
        print(
            check_out(
                f"output_files/seq/output{test}.txt",
                f"output_files/mpi/output{test}.txt",
            )
        )
