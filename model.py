import torch
from seq2seq import Seq2Seq
from encoder import Encoder
from decoder import Decoder
from tokenize import tokenize, untokenize
import keyword


def augment_tokenize_python_code(python_code_str, mask_factor=0.3):

    var_dict = {}  # Dictionary that stores masked variables

    # certain reserved words that should not be treated as normal variables and
    # hence need to be skipped from our variable mask augmentations
    skip_list = ['range', 'enumerate', 'print', 'ord', 'int', 'float', 'zip'
                 'char', 'list', 'dict', 'tuple', 'set', 'len', 'sum', 'min', 'max']
    skip_list.extend(keyword.kwlist)

    var_counter = 1
    python_tokens = list(
        tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_output = []

    for i in range(0, len(python_tokens)):
        if python_tokens[i].type == 1 and python_tokens[i].string not in skip_list:

            # avoid masking modules, functions and error literals
            if i > 0 and python_tokens[i-1].string in ['def', '.', 'import', 'raise', 'except', 'class']:
                skip_list.append(python_tokens[i].string)
                tokenized_output.append(
                    (python_tokens[i].type, python_tokens[i].string))
            elif python_tokens[i].string in var_dict:  # if variable is already masked
                tokenized_output.append(
                    (python_tokens[i].type, var_dict[python_tokens[i].string]))
            elif random.uniform(0, 1) > 1-mask_factor:  # randomly mask variables
                var_dict[python_tokens[i].string] = 'var_' + str(var_counter)
                var_counter += 1
                tokenized_output.append(
                    (python_tokens[i].type, var_dict[python_tokens[i].string]))
            else:
                skip_list.append(python_tokens[i].string)
                tokenized_output.append(
                    (python_tokens[i].type, python_tokens[i].string))

        else:
            tokenized_output.append(
                (python_tokens[i].type, python_tokens[i].string))

    return tokenized_output


class MyModel():
    def __init__(self, model_path, src_path, trg_path):
        self.Input = torch.load(src_path)
        self.Output = torch.load(trg_path)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        INPUT_DIM = len(self.Input.vocab)
        OUTPUT_DIM = len(self.Output.vocab)
        HID_DIM = 256
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 16
        DEC_HEADS = 16
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        enc = Encoder(INPUT_DIM,
                      HID_DIM,
                      ENC_LAYERS,
                      ENC_HEADS,
                      ENC_PF_DIM,
                      ENC_DROPOUT,
                      self.device)

        dec = Decoder(OUTPUT_DIM,
                      HID_DIM,
                      DEC_LAYERS,
                      DEC_HEADS,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      self.device)

        SRC_PAD_IDX = self.Input.vocab.stoi[self.Input.pad_token]
        TRG_PAD_IDX = self.Output.vocab.stoi[self.Output.pad_token]

        self.model = Seq2Seq(enc, dec, SRC_PAD_IDX,
                             TRG_PAD_IDX, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path))

        from deep_translator import GoogleTranslator
        self.helper = GoogleTranslator(source="vietnamese", target="english")

    def predict(self, src):
        src = self.helper.translate(src)
        src = src.split(" ")
        translation, attention = self.__translate_sentence(
            src, self.Input, self.Output, self.model, self.device)

        return untokenize(translation[:-1]).decode('utf-8')

    def __translate_sentence(self, sentence, src_field, trg_field, model, device, max_len=50000):

        model.eval()

        if isinstance(sentence, str):
            nlp = spacy.load('en')
            tokens = [token.text.lower() for token in nlp(sentence)]
        else:
            tokens = [token.lower() for token in sentence]

        tokens = [src_field.init_token] + tokens + [src_field.eos_token]

        src_indexes = [src_field.vocab.stoi[token] for token in tokens]

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

        src_mask = model.make_src_mask(src_tensor)

        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_mask)

        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

        for i in range(max_len):

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

            trg_mask = model.make_trg_mask(trg_tensor)

            with torch.no_grad():
                output, attention = model.decoder(
                    trg_tensor, enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:, -1].item()

            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break

        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

        return trg_tokens[1:], attention
