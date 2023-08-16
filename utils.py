import numpy as np

def find_quantiles(documents, tokenizer):
    num_tokens_per_doc = [
        len(tokenizer((doc))['input_ids']) 
            for doc in documents
    ]
    num_tokens_per_doc_array = np.array(num_tokens_per_doc)

    quantiles_to_calc = [
        i for i in np.linspace(0, 1.0, num=11)
    ]
    quantiles = {
        quant : int(np.quantile(num_tokens_per_doc_array, quant))
            for quant in quantiles_to_calc
    }

    return quantiles