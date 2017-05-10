import pandas as pd 
import numpy as np
import nltk
import operator
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold
import sys


#print "Loading DF"
df = pd.read_csv("totalGAOPOSTagged.csv")
#print "Done loading DF"

lowerCaser = lambda x: str(x).lower()

df['sentence'] = df['sentence'].map(lowerCaser)

def onlySecondPartsOfTuples(tupleList):
	newList = []
	for t in tupleList:
		newList.append(t[1])

	return newList



#posTaggerLambda = lambda x: ' '.join(onlySecondPartsOfTuples(nltk.pos_tag(nltk.word_tokenize(x))))

#print "POS TAGGING SENTENCE"
#df['pos_tag'] = df['sentence'].map(posTaggerLambda)

#df.to_csv("totalGAOPOSTagged.csv")

#sys.exit()

#print df.head

#nltk.word_tokenize(...)
#nltk.pos_tag(array)

currentWord = ""

def checkIfCurrentWordInSentence(sentence):
	if currentWord in sentence:
		return True
	else:
		return False


currentWordLambda = lambda x: checkIfCurrentWordInSentence(x)

relevant_cols = []

Unigrams = {}
Bigrams = {}

POSUnigrams = {}
POSBigrams = {}



#print "loading Unigrams and bigrams and POS unigrams and Bigrams"

for index, row in df.iterrows():
	#if index % 100 == 0:
		#print index
	tokens = nltk.word_tokenize(row['sentence'])
	POStokens = row['pos_tag'].split(" ")
	for i in range(0, len(tokens)):
		
		word = tokens[i]
		POS = POStokens[i]


		if POS in POSUnigrams:
			POSUnigrams[POS] += 1
		else:
			POSUnigrams[POS] = 1

		if word in Unigrams:
			Unigrams[word] += 1
		else:
			Unigrams[word] = 1

		if i < len(tokens) - 1:

			POSbigram = POS + " " + POStokens[i + 1]
			bigram = word + " " + tokens[i + 1]
			if bigram in Bigrams:
				Bigrams[bigram] += 1
			else:
				Bigrams[bigram] = 1

			if POSbigram in POSBigrams:
				POSBigrams[POSbigram] += 1
			else:
				POSBigrams[POSbigram] = 1

numFeatures = 1000

#print sorted(Unigrams.items(), key=operator.itemgetter(1))
#print "Making unigram features"
for key, value in sorted(Unigrams.items(), key=operator.itemgetter(1))[-1 * numFeatures:]:
	#if i %100 == 0:
		#print i
	#print key
	currentWord = key
	df['has_' + key] = df['sentence'].map(checkIfCurrentWordInSentence)
	relevant_cols.append('has_' + key)
	i += 1

i = 0

#print "Making bigram features"
for key, value in sorted(Bigrams.items(), key=operator.itemgetter(1))[-1 * numFeatures:]:
	#if i %100 == 0:
		#print i
	#print key+ str(value)
	currentWord = key
	df['has_' + key] = df['sentence'].map(checkIfCurrentWordInSentence)
	relevant_cols.append('has_' + key)
	i += 1


#rint sorted(Unigrams.items(), key=operator.itemgetter(1))
#print "Making unigram POS features"
for key, value in sorted(POSUnigrams.items(), key=operator.itemgetter(1))[-1 * numFeatures:]:
	#if i %100 == 0:
		#print i
	##print key
	currentWord = key
	df['hasPOS_' + key] = df['pos_tag'].map(checkIfCurrentWordInSentence)
	relevant_cols.append('hasPOS_' + key)
	i += 1

#i = 0

#print sorted(POSUnigrams.items(), key=operator.itemgetter(1))

#print "Making bigram POS features"
for key, value in sorted(POSBigrams.items(), key=operator.itemgetter(1))[-1 * numFeatures:]:
	#if i %100 == 0:
		#print i
	#print key+ str(value)
	currentWord = key
	df['hasPOS_' + key] = df['pos_tag'].map(checkIfCurrentWordInSentence)
	relevant_cols.append('hasPOS_' + key)
	i += 1


#print sorted(POSBigrams.items(), key=operator.itemgetter(1))


#sys.exit()




#print df.shape
#print(len(df))

attributes = df[relevant_cols].as_matrix()
classes = df['class'].as_matrix()
#print(attributes[:10])



sel = VarianceThreshold(threshold=(.99*(1-.99)))
newAtts = sel.fit_transform(attributes)
#print "WOW"
#print len(attributes[0])
#print len(newAtts[0])


#print(df['class'].value_counts())

#sys.exit()

#attributes = newAtts





crossValSize = 10; 

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)

#scores = cross_val_score(clf, attributes, classes, cv = crossValSize)

#print("MLP Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#gnb = GaussianNB()

#y_pred = gnb.fit(attributes, classes).predict(attributes)

#scores = cross_val_score(gnb, attributes, classes, cv = crossValSize)

#print("Gaussian NaiveBayes Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





	

mnb = MultinomialNB()

y_pred = mnb.fit(attributes, classes).predict(attributes)

scores = cross_val_score(mnb, attributes, classes, cv = crossValSize)

#print("Multinomial NaiveBayes Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#print("hi: " )
#print(len(mnb.feature_log_prob_))


classOne = mnb.feature_log_prob_[1]
classZero = mnb.feature_log_prob_[0]

maxIndexZero = np.argmax(classZero)
minIndexZero = np.argmin(classZero)

maxIndexOne = np.argmax(classOne)
minIndexOne = np.argmin(classOne)



zeroMins = []
zeroMaxs = []

topAmount = 100


topProbsIndexesZero = np.argpartition(classZero, -1 * topAmount)[-1 * topAmount:]


#print(topProbsIndexesZero)

topProbsZero = []



for index in topProbsIndexesZero:
	topProbsZero.append(classZero[index])

#print(topProbsZero)

#print("SECOND MAX: ")
#print(secondMaxIndexZero)
#print("REGULAR MAX: " + str(maxIndexZero))

for i in range(0, len(classZero)):
	currentProb = classZero[i]

	if(currentProb == classZero[minIndexZero]):
		zeroMins.append(i)
	if(currentProb in topProbsZero):
		zeroMaxs.append(i)

#print("ZERO MINS LENGTH: " + str(len(zeroMins)))


#for index in zeroMins:
	#print(relevant_cols[index])

#for index in zeroMaxs:
	#print(relevant_cols[index])




oneMins = []
oneMaxs = []

topProbsIndexesOne = np.argpartition(classOne, -1 * topAmount)[-1 * topAmount:]

#print(topProbsIndexesOne)

topProbsOne= []



for index in topProbsIndexesOne:
	topProbsOne.append(classOne[index])

#print(topProbsOne)

#print("SECOND MAX: ")
#print(secondMaxIndexZero)
#print("REGULAR MAX: " + str(maxIndexZero))

for i in range(0, len(classOne)):
	currentProb = classOne[i]

	if(currentProb == classOne[minIndexOne]):
		oneMins.append(i)
	if(currentProb in topProbsOne):
		oneMaxs.append(i)

#print("ZERO MINS LENGTH: " + str(len(zeroMins)))


#for index in oneMins:
	#print(relevant_cols[index])


#sys.exit()

#for index in oneMaxs:
	#print(relevant_cols[index])



#zeroMaxFeatures = []
#oneMaxFeatures = []


zeroMaxsPairs = {}
oneMaxsPairs = {}

for index in zeroMaxs:
	if index not in oneMaxs:
		zeroMaxsPairs[relevant_cols[index]] = classZero[index]

for index in oneMaxs:
	
	oneMaxsPairs[relevant_cols[index]] = classOne[index]





print("HERES ZERO: ")

for k,v in sorted(zeroMaxsPairs.items(), key=lambda p: p[1], reverse=True):
	#if k not in oneMaxsPairs:
	print(k, v)


print("HERES ONES: ")

for k,v in sorted(oneMaxsPairs.items(), key=lambda p: p[1], reverse=True):
	#if k not in oneMaxsPairs:
	print(k, v)


#print("LENGTH: " + str(len(oneMins)))


#for index in zeroMins:
	#if index not in oneMins:
		#print(relevant_cols[index])






#for index in zeroMaxs:
	#if index not in oneMaxs:
		#print(relevant_cols[index])
