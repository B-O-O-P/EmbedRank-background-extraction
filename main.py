import argparse
import os
import sys
import nltk
import subprocess
import shlex
from swisscom import launch

sys.path.append("..")


def start_standford_server():
    current_dir = str(os.path.dirname(os.path.realpath(__file__)))
    stanford_server_command = shlex.split(
        "java -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,"
        "pos -status_port 9000 -port 9000 -timeout 15000 & ")
    FNULL = open(os.devnull, 'w')
    return subprocess.Popen(stanford_server_command, stdout=FNULL, stderr=FNULL,
                            cwd="{}/stanford-corenlp-full-2018-02-27".format(current_dir))


def extract_location(embedding_model, model_path, text):
    embedding_distributor = launch.load_local_embedding_distributor(embedding_model, model_path)
    pos_tagger = launch.load_local_corenlp_pos_tagger()
    kp = launch.extract_keyphrases(embedding_distributor, pos_tagger, text, N=3, lang="en", alias_threshold=0.1)
    locations, relevances, aliases = kp
    return locations


def predict_sent2vec_location(text):
    current_dir = str(os.path.dirname(os.path.realpath(__file__)))
    data_path = "{}/data/wiki_bigrams.bin".format(current_dir)
    locations = extract_location("sent2vec", data_path, text)
    return locations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract location from raw text")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-raw_text", help="raw text to process")

    args = parser.parse_args()
    raw_text = args.raw_text

    nltk.download("punkt")

    text = raw_text
    print(predict_sent2vec_location(text))
