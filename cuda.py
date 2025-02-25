# Funzione che controlla riga per riga se due file sono uguali
def check_out(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        for line1, line2 in zip(f1, f2):
            if line1 != line2:
                return False
    return True

if __name__ == "__main__":
    inpt = input("Inserisci il test: ")
    print(check_out(f"output_files/seq/output{inpt}.txt", "output_cuda.txt"))
