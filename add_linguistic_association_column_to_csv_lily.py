from sys import argv

from ldm.model.ngram import PPMINgramModel
from ldm.preferences.preferences import Preferences
from ldm.utils.exceptions import WordNotFoundError


def main(in_path: str, out_path: str):
    corpus = Preferences.source_corpus_metas.ukwac
    model = PPMINgramModel(
        corpus_meta=corpus,
        window_radius=5)
    model.train(memory_map=True)
    with open(in_path, mode="r", encoding="utf-8") as csv_in_file:
        with open(out_path, mode="w", encoding="utf-8") as csv_out_file:
            header_line = csv_in_file.readline().strip()
            csv_out_file.write(f"{header_line},Linguistic association: {model.name.replace(',', '')}\n")
            for line in csv_in_file:
                line = line.strip()
                word_1: str
                word_2: str
                try:
                    word_1, word_2, *_others = line.split(",")
                    word_1 = word_1.strip().lower()
                    word_2 = word_2.strip().lower()
                    association = model.association_between(word_1, word_2)
                except WordNotFoundError:
                    association = ""
                csv_out_file.write(f"{line},{association}\n")


if __name__ == '__main__':
    print("Running %s..." % " ".join(argv))
    main(argv[1], argv[2])
    print("Done!")
