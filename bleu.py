import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction


def remove_special(hyp):
	return hyp.replace("<sos>", "").replace("<eos>", "").replace("<pad>", "").replace("UNK", "").replace("unk", "")


def fetch_bleu_score(refs, hyp, return_hyp=False):
	chencherry = SmoothingFunction()
	tokenized_refs = []
	for ref in refs:
		tokenized_refs.append(word_tokenize(ref))
	clean_hyp = remove_special(hyp)
	tokenized_hyp = word_tokenize(clean_hyp)
	# bleu = nltk.translate.bleu_score.sentence_bleu(tokenized_refs, tokenized_hyp, weights=(0.1666, 0.1666, 0.1666, 0.1666, 0.1666))
	bleu = nltk.translate.bleu_score.sentence_bleu(tokenized_refs, tokenized_hyp)
	if return_hyp:
		return bleu, clean_hyp
	else:
		return bleu
