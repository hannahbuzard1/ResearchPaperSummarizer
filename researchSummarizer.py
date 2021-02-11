#Hannah and Merril
#Final Project CSCI404
#Spring 2020

from difflib import SequenceMatcher #used for Rouge-l
import random
import os
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer #used for tf-idf vectorization
from nltk.tokenize import RegexpTokenizer
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#function to print out average recall,precision, and f1 scores
def printAverages(avglist) :
    sum = 0
    for element in avglist:
        sum += element
    return sum/len(avglist)
#arrays that hold the recall, precision, and f1 f1scores1
#for every document
precisions1 = []
precisions2 = []
precisions3 = []
recalls1 = []
recalls2 = []
recalls3 = []
f1scores1 = []
f1scores2 = []
f1scores3 = []
#for every file in the directory of training files, we need to generate a summary
directory = os.listdir(r"C:\Users\buzardh\Desktop\FinalProject\Trainfiles")
for file in directory :
    filename = os.path.splitext(file)[0]
    file = open(file, encoding="utf8" )
    text = file.read()
    ##prepping for stop word removal
    stop_words = set(stopwords.words('english'))
    tokens = sent_tokenize(text)
    tokens2 = []
    #remove ']' and '[' because they were skewing data
    for sentence in tokens:
        if "[" in sentence:
            sentence = sentence.replace("[", "")
        if "]" in sentence:
            sentence = sentence.replace("]", "")
        tokens2.append(sentence)
    #stop word removal
    withoutstops = []
    for w in tokens2:
        text_tokens = word_tokenize(w)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        filtered_sentence = (" ").join(tokens_without_sw)
        withoutstops.append(filtered_sentence)
    #remove newline characters
    processed = []
    for sentence in withoutstops :
        string = []
        for word in sentence:
            if "\n" in word:
                word = word.replace("\n", " ")
            string.append(word)
        processed.append(string)
    #remove punctation
    tokenizer = RegexpTokenizer(r'\w+')
    nopunct = []
    for sentences in processed:
        sent = ''.join(sentences)
        string = tokenizer.tokenize(sent)
        nopunct.append(string)
    #stem remaining words
    ps = PorterStemmer()
    stemmedwords = []
    for sentences in nopunct:
        string = []
        for word in sentences:
            string.append(ps.stem(word).lower())
        stemmedwords.append(string)
    ##vectorization of the sentences using tf-idf scores
    text = []
    for sentence in stemmedwords:
        sentence = ' '.join(sentence)
        text.append(sentence)
    tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    tfidf_matrix = tfidf.fit_transform(text)
    matrix = tfidf_matrix.toarray()
    ##generate sentence scores for each sentence
    allscores = []
    count = 0
    for sentence in matrix:
        sentscore = 1
        for score in sentence:
            if score != 0:
                sentscore = sentscore * score
        sen = tokens[count].replace("\n", " ")
        arr = [sen, sentscore]
        allscores.append(arr)
        count+=1
    ##get highest scoring sentences (sort in descending order)
    allscores.sort(key=lambda x:x[1])
    ##read in test data
    testfile = str(filename) + "Test" + ".txt"
    testfile = open(testfile, encoding="utf8" )
    testtext = testfile.read()
    #get size of test data
    testsize = len(sent_tokenize(testtext))
    summary = []
    #size of generated summary should be the same size as the test abstract
    #add top scoring sentences in range of size
    size = testsize
    for i in range (0,size) :
        summary.append(allscores[i][0])
    print(summary)
    #calculate accuracy for unigrams using ROUGE scores
    #add each score to it's corresponding global list
    counter = 0
    testgrams = list(ngrams(testtext,1))
    summary = " ".join(summary)
    traingrams = list(ngrams(summary,1))
    testlen = len(testgrams)
    trainlen = len(traingrams)
    for gram in testgrams :
        if gram in traingrams:
            counter+=1
    print("Unigrams Recall:")
    recall = counter/testlen
    print(recall)
    recalls1.append(recall)
    print("Unigrams Precision:")
    precision = counter/trainlen
    print(precision)
    precisions1.append(precision)
    print("Unigrams F1 Score:")
    f1score =(2 * precision * recall)/(precision + recall)
    print(f1score)
    f1scores1.append(f1score)
    #calculate accuracy for bigrams using ROUGE scores
    #add each score to it's corresponding global list
    counter2 = 0
    testgrams2 = list(ngrams(testtext,2))
    traingrams2 = list(ngrams(summary,2))
    testlen2 = len(testgrams2)
    trainlen2 = len(traingrams2)
    for gram in testgrams2 :
        if gram in traingrams2:
            counter2+=1
    print("Bigrams Recall:")
    recall = counter2/testlen2
    recalls2.append(recall)
    print(recall)
    print("Bigrams Precision:")
    precision = counter2/trainlen2
    precisions2.append(precision)
    print(precision)
    print("Bigrams F1 Score:")
    f1score =(2 * precision * recall)/(precision + recall)
    print(f1score)
    f1scores2.append(f1score)
    #calculate accuracy for trigrams using ROUGE scores
    #add each score to it's corresponding global list
    counter3 = 0
    testgrams3 = list(ngrams(testtext,3))
    traingrams3 = list(ngrams(summary,3))
    testlen3 = len(testgrams3)
    trainlen3 = len(traingrams3)
    for gram in testgrams3 :
        if gram in traingrams3:
            counter3+=1
    print("Trigrams Recall:")
    recall = counter3/testlen3
    recalls3.append(recall)
    print(recall)
    print("Trigrams Precision:")
    precision = counter3/trainlen3
    print(precision)
    precisions3.append(precision)
    print("Trigrams F1 Score:")
    f1score =(2 * precision * recall)/(precision + recall)
    print(f1score)
    f1scores3.append(f1score)
    #Calculate Rouge-l - longest common subsequence between summaries
    #this is not included in our results but we wanted to include it
    #because we were interested in what the scores would be
    print("Rouge-l")
    seqMatch = SequenceMatcher(None,summary,testtext, autojunk = False)
    match = seqMatch.find_longest_match(0, len(summary), 0, len(testtext))
    print (summary[match.a: match.a + match.size])
    arr = word_tokenize(summary[match.a: match.a + match.size])
    print(len(arr))
    gramsize = len(arr)
    #if the longest subsequence is larger than a trigram (because we already calculated tp to trigram accuracy)
    #then calculate recall, precision, and f1-scores using the length of the longest common subsequence
    #again, this is not included in our results...
    if gramsize > 3 :
        counter4 = 0
        testgrams4 = list(ngrams(testtext,gramsize))
        traingrams4 = list(ngrams(summary,gramsize))
        testlen4 = len(testgrams4)
        trainlen4 = len(traingrams4)
        for gram in testgrams4 :
            if gram in traingrams4:
                counter4+=1
        print("Rouge-L Recall:")
        recall = counter4/testlen4
        print(recall)
        print("Rouge-L Precision:")
        precision = counter4/trainlen4
        print(precision)
        print("Rouge-L F1 Score:")
        f1score =(2 * precision * recall)/(precision + recall)
        print(f1score)
#use printAverages function to get average unigram, bigram, and trigram scores for recall, precision, and f1-scores
#and print out these averages
print("Average unigram recall: ")
print(printAverages(recalls1))
print("Average bigram recall: ")
print(printAverages(recalls2))
print("Average trigram recall: ")
print(printAverages(recalls3))
print("Average unigram precision: ")
print(printAverages(precisions1))
print("Average bigram precision: ")
print(printAverages(precisions2))
print("Average trigram precision: ")
print(printAverages(precisions3))
print("Average unigram f1-score: ")
print(printAverages(f1scores1))
print("Average bigram f1-score: ")
print(printAverages(f1scores2))
print("Average trigram f1-score: ")
print(printAverages(f1scores3))
