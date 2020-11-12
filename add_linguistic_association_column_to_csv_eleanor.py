from sys import argv
from functools import partial

from pandas import DataFrame, read_csv

from ldm.model.ngram import PPMINgramModel
from ldm.preferences.preferences import Preferences
from ldm.utils.exceptions import WordNotFoundError


def get_association(row, model):
    word_1 = row["Word"].lower().strip()
    word_2 = row["Colour"].lower().strip()
    try:
        return model.association_between(word_1, word_2)
    except WordNotFoundError:
        return None


def main(in_path: str, out_path: str):
    model = PPMINgramModel(
        corpus_meta=Preferences.source_corpus_metas.ukwac,
        window_radius=5)
    model.train(memory_map=True)
    with open(in_path, mode="r", encoding="utf-8") as csv_in_file:
        df: DataFrame = read_csv(csv_in_file, header=0, index_col=None)
    df["Linguistic Co-occurrence Value"] = df[["Word", "Colour"]].apply(partial(get_association, model=model), axis=1)
    with open(out_path, mode="w", encoding="utf-8") as csv_out_file:
        df.to_csv(csv_out_file, index=False)


if __name__ == '__main__':
    print("Running %s..." % " ".join(argv))
    main(argv[1], argv[2])
    print("Done!")
