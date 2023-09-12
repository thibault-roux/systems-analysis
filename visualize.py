import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_dict(system):
    ind2tok = dict()
    tok2ind = dict()
    ind = 0
    if "bpe" in system:
        type_tok = "bpe"
        number = system.split("bpe")[1].split("_")[0]
    else:
        type_tok = "char"
        number = "100"
    if number == "50":
        number = "70"
    with open("../results_lia_asr/" + system + "/1234/save/" + number + "_" + type_tok + ".vocab", "r") as f:
        for line in f:
            tok = line.split()[0]
            tok = tok.replace("▁", " ")
            ind2tok[ind] = tok
            tok2ind[tok] = ind
            ind += 1
    return ind2tok, tok2ind


def plot_ctc_heatmap(system, vocab, normtype): # vocab = {token, char, all}, normtype = {exp, norm}
    tensor = pickle.load(open("pickle2/tensor_" + system + ".pkl", "rb"))

    # sentence = " alors nous avons un"
    sentence = " à la mémoire d' un général battu monsieur bruno le maire"

    # Load dict
    ind2tok, tok2ind = load_dict(system)
    if vocab == "char":
        # useful_toks = [' ', '<unk>', 'a', 'l', 'o', 'r', 's', 'n', 'u', 'v']
        useful_toks = list(set(sentence))
        useful_toks.append("<unk>")
    elif vocab == "token":
        # useful_toks = [' ', ' a', '<unk>', 'a', 'al', ' alors', 'l', 'lo', 'o', 'or', 'ors', 'r', 's']
        # monsieur mémoire
        useful_toks = [' ', ' m', ' mo', ' mon', ' monsieur', ' mé', '<unk>', 'e', 'eur', 'i', 'ie', 'ir', 'ire', 'm', 'mo', 'mé', 'n', 'né', 'o', 'oi', 'oir', 'oire', 'on', 'ons', 'r', 're', 's', 'si', 'sie', 'sieur', 'u', 'ur', 'é', 'én', 'ér']
    elif vocab == "all":
        useful_toks = set()
        useful_toks.add("<unk>")
        for ind, tok in ind2tok.items():
            if tok in sentence:
                useful_toks.add(tok)
        useful_toks = list(useful_toks)
    elif vocab == "partial":
        useful_toks = []
        # keep the rows where there is at least one value > 0.1
        tensor2 = np.exp(tensor)
        for i in range(tensor2.shape[1]):
            if np.max(tensor2[:, i]) > 0.1:
                useful_toks.append(ind2tok[i])
    else:
        raise ValueError("vocab should be 'token' or 'char' or 'all'")
    useful_toks.sort()
    useful_rows_ids = [tok2ind[tok] for tok in useful_toks if tok in tok2ind.keys()]
    if len(useful_rows_ids) < 3:
        print("Not enough tokens in the sentence")
        return

    # normalization
    if normtype == "exp":
        # compute the exponential of the tensor
        tensor = np.exp(tensor)
    elif normtype == "norm":
        min_values = np.min(tensor, axis=1)
        max_values = np.max(tensor, axis=1)
        tensor = (tensor - min_values[:, np.newaxis]) / (max_values[:, np.newaxis] - min_values[:, np.newaxis])
    else:
        raise ValueError("normtype should be 'exp' or 'norm'")
    # else: tensor = tensor / np.linalg.norm(tensor, axis=1, keepdims=True)
    

    # keep only the rows with interesting tokens
    filtered_tensor = tensor[:, useful_rows_ids]

    # transpose to have temporality in x-axis
    filtered_tensor = np.transpose(filtered_tensor)
    # Create a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(filtered_tensor, cmap='viridis', aspect='auto')
    # plt.title("CTC Layer Output Heatmap")
    # plt.xlabel("Token Index")
    # plt.ylabel("Speech Segment")
    plt.yticks(np.arange(len(useful_toks)), useful_toks)
    plt.colorbar(label="Value")
    plt.show()

    plt.savefig("figures2/" + vocab + "/" + normtype + "/heatmap_" + system + ".png")
    plt.close('all')

if __name__ == "__main__":
    systems = ["wav2vec2_ctc_fr_1k", "wav2vec2_ctc_fr_3k", "wav2vec2_ctc_fr_7k", "wav2vec2_ctc_fr_bpe1000_7k", "wav2vec2_ctc_fr_bpe100_7k", "wav2vec2_ctc_fr_bpe1500_7k", "wav2vec2_ctc_fr_bpe150_7k", "wav2vec2_ctc_fr_bpe250_7k", "wav2vec2_ctc_fr_bpe500_7k", "wav2vec2_ctc_fr_bpe50_7k", "wav2vec2_ctc_fr_bpe650_7k", "wav2vec2_ctc_fr_bpe750_7k", "wav2vec2_ctc_fr_bpe900_7k", "wav2vec2_ctc_fr_xlsr_53_french", "wav2vec2_ctc_fr_xlsr_53"] 
    for system in systems:
        # plot_ctc_heatmap("wav2vec2_ctc_fr_bpe1500_7k", "all", "norm")
        print(system)
        for token in ["partial"]: # ["token", "char", "all", "partial"]:
            for norm in ["exp", "norm"]:
               plot_ctc_heatmap(system, token, norm) # vocab = {token, char, all}, normtype = {exp, norm}