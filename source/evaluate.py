import argparse, sys, torch
sys.path.append('/home/liuxg/workspace/SAparaphrase/')
sys.path.append('/home/liuxg/workspace/SAparaphrase/bert')
from bert.bertinterface import BertEncoding, BertSimilarity
from utils import get_corpus_bleu_scores, savetexts

class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):

                f.write("%s, %s\n" % (key, str(value)))

def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--sim', default='word_max', type=str)
    parser.add_argument('--mode', default='bleu', type=str)
    parser.add_argument('--reference_path', default=None, type=str)
    parser.add_argument('--generated_path', default=None, type=str)

    d = vars(parser.parse_args())
    option = Option(d)
    if option.mode == 'bleu':
        evaluate_bleu(option.reference_path, option.generated_path)
    elif option.mode =='semantic':
        evaluate_semantic(option.reference_path, option.generated_path)
    else:
        pass

 

def evaluate_bleu(reference_path, generated_path):
    # Evaluate model scores
    actual_word_lists = []
    with open(reference_path) as f:
        for line in f:
            actual_word_lists.append([line.strip().lower().split()])

    generated_word_lists = []
    with open(generated_path) as f:
        for line in f:
            generated_word_lists.append(line.strip().lower().split())

    bleu_scores = get_corpus_bleu_scores(actual_word_lists, generated_word_lists)
    print('bleu scores:', bleu_scores)

def evaluate_semantic(reference_path, generated_path):
    actual_word_lists = []
    with open(reference_path) as f:
        for line in f:
            actual_word_lists.append(line.strip())

    generated_word_lists = []
    with open(generated_path) as f:
        for line in f:
            generated_word_lists.append(line.strip())


    model =  BertEncoding()
    rep1s, rep2s = [],[]
    batchsize =int(100)
    for i in range(int(len(generated_word_lists)/batchsize)):
        s1s = actual_word_lists[i*batchsize:i*batchsize+batchsize]
        s2s = generated_word_lists[i*batchsize:i*batchsize+batchsize]
        rep1 = model.get_encoding(s1s)
        rep1s.append(rep1)
        rep2 = model.get_encoding(s2s)
        rep2s.append(rep2)
    rep1 = torch.cat(rep1s,0) 
    rep2 = torch.cat(rep2s,0) 
    summation = torch.sum(rep1*rep2,1)/(rep1.norm(2,1)*rep2.norm(2,1))
    print(torch.mean(summation))



def test_semantic(s1, s2):
    model =  BertSimilarity()
    rep1 = model.get_encoding(s1,s1)
    rep2 = model.get_encoding(s1,s2)
    rep3 = model.get_encoding(s2,s2)
    rep1 = (rep1+rep3)/2
    semantic = torch.sum(rep1*rep2,1)/(rep1.norm()*rep2.norm())
    semantic = semantic*(1- (abs(rep1.norm()-rep2.norm())/max(rep1.norm(),rep2.norm())))
    print(torch.mean(semantic))



if __name__ == "__main__":
    main()
    s1 = ['do you like the red car']
    s2 = ['is the red car  your favorite']
    s3 = ['how should i prepare for lunch']
    s4 = ['how should you prepare for gpa ']
    s5 = ['how should you make for lunch']
    # test_semantic(s1,s2)
    # test_semantic(s4,s2)
    # test_semantic(s4,s3)
    # test_semantic(s5,s3)

