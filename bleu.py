# n-gram 분석 https://blog.ilkyu.kr/entry/%EC%96%B8%EC%96%B4-%EB%AA%A8%EB%8D%B8%EB%A7%81-ngram


def word_ngram(sentence, num_gram):
    # sentence: 분석할 문장, num_gram: n-gram 단위
    # in the case a file is given, remove escape characters
    sentence = sentence.replace('\n', ' ').replace('\r', ' ')
    text = tuple(sentence.split(' '))
    ngrams = [text[x:x + num_gram] for x in range(0, len(text)) if x + num_gram <= len(text)]
    return list(ngrams)


# n-gram 빈도 리스트 생성
def make_freqlist(ngrams):
    unique_ngrams = list(set(ngrams))
    freqlist = [0 for _ in range(len(unique_ngrams))]
    for ngram in ngrams:
        idx = unique_ngrams.index(ngram)
        freqlist[idx] += 1
    result = [unique_ngrams, freqlist]
    return result


# 두개 ngram 얼마나 겹치는지
def precision(output, target):  #
    result = 0
    for i in range(len(output[0])):
        if output[0][i] in target[0]:
            idx = target[0].index(output[0][i])
            result += min(output[1][i], target[1][idx])
    return result / sum(output[1])


def n_gram_precision(sen_out, sen_tar):
    output = []
    target = []
    for i in range(1, 5):
        n_gram = word_ngram(sen_out, i)
        out_tmp = make_freqlist(n_gram)
        output.append(out_tmp)
        n_gram2 = word_ngram(sen_tar, i)
        tar_tmp = make_freqlist(n_gram2)
        target.append(tar_tmp)
    result = 0
    for i in range(len(output)):
        n_pre = precision(output[i], target[i])
        if i == 0:
            result = n_pre
        else:
            result *= n_pre
    # Brevity Penalty
    result = pow(result, 1 / 4)
    bp = min(1, sum(output[0][1]) / sum(target[0][1]))
    return bp * result
