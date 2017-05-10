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
from sklearn.metrics import confusion_matrix, accuracy_score
import sys



#print "Loading DF"
df = pd.read_csv("totalGAOPOSTTaggedRanked.csv")
#print "Done loading DF"

lowerCaser = lambda x: str(x).lower()

df['sentence'] = df['sentence'].map(lowerCaser)

def onlySecondPartsOfTuples(tupleList):
	newList = []
	for t in tupleList:
		newList.append(t[1])

	return newList



def squash(x):
	if(x < 3):
		return 1
	elif(x >3):
		return 5
	else:
		return x

squasher = lambda x : squash(x)

df['ranking'] = df['ranking'].map(squasher)
print(df['ranking'])

print("HERE: ")
print(df['ranking'].value_counts())

sys.exit()

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
	#print(row['sentence'])
	#if index % 100 == 0:
		#print index
	tokens = nltk.word_tokenize(row['sentence'])
	POStokens = row['pos_tag'].split(" ")
	for i in range(0, len(tokens)):
		
		word = tokens[i]

		POS = ""

		if(i < len(POStokens)):
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
			if i < len(POStokens) -1:
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
print "Making unigram features"
for key, value in sorted(Unigrams.items(), key=operator.itemgetter(1))[-1 * numFeatures:]:
	if i %100 == 0:
		print i
	#print key
	currentWord = key
	df['has_' + key] = df['sentence'].map(checkIfCurrentWordInSentence)
	relevant_cols.append('has_' + key)
	i += 1

i = 0

print "Making bigram features"
for key, value in sorted(Bigrams.items(), key=operator.itemgetter(1))[-1 * numFeatures:]:
	if i %100 == 0:
		print i
	#print key+ str(value)
	currentWord = key
	df['has_' + key] = df['sentence'].map(checkIfCurrentWordInSentence)
	relevant_cols.append('has_' + key)
	i += 1


#rint sorted(Unigrams.items(), key=operator.itemgetter(1))
print "Making unigram POS features"
for key, value in sorted(POSUnigrams.items(), key=operator.itemgetter(1))[-1 * numFeatures:]:
	#if i %100 == 0:
		#print i
	##print key
	currentWord = key
	df['hasPOS_' + key] = df['pos_tag'].map(checkIfCurrentWordInSentence)
	relevant_cols.append('hasPOS_' + key)
	#i += 1

#i = 0

#print sorted(POSUnigrams.items(), key=operator.itemgetter(1))

print "Making bigram POS features"
for key, value in sorted(POSBigrams.items(), key=operator.itemgetter(1))[-1 * numFeatures:]:
	if i %100 == 0:
		print i
	#print key+ str(value)
	currentWord = key
	df['hasPOS_' + key] = df['pos_tag'].map(checkIfCurrentWordInSentence)
	relevant_cols.append('hasPOS_' + key)
	i += 1


#relevant_cols.append("language_model_score")


#print sorted(POSBigrams.items(), key=operator.itemgetter(1))


#sys.exit()




#print df.shape
#print(len(df))

attributes = df[relevant_cols].as_matrix()
#classes = df['class'].as_matrix()
classes = df['ranking'].as_matrix()
#print(attributes[:10])



sel = VarianceThreshold(threshold=(.99*(1-.99)))
newAtts = sel.fit_transform(attributes)
#print "WOW"
#print len(attributes[0])
#print len(newAtts[0])


#print(df['class'].value_counts())

#sys.exit()

#attributes = newAtts







crossValSize = 5; 

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
#scores = cross_val_score(clf, attributes, classes, cv = crossValSize)

#print("MLP Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

gnb = GaussianNB()

y_pred = gnb.fit(attributes[:300], classes[:300]).predict(attributes[300:])

scores = cross_val_score(gnb, attributes, classes, cv = crossValSize)

print("Gaussian NaiveBayes Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print("CONFUSION MATRIX: ")
print(confusion_matrix(classes[300:], y_pred))
print(accuracy_score(classes[300:], y_pred))

#print("Number of mislabeled points out of a total %d points : %d" % (400,(classes[300:] != y_pred).sum()))

	
sys.exit()


mnb = MultinomialNB()

y_pred = mnb.fit(attributes, classes).predict(attributes)
scores = cross_val_score(mnb, attributes, classes, cv = crossValSize)

print("Multinomial NB Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, attributes, classes, cv = crossValSize)

print("Linear Kernel Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
y_pred = clf.fit(attributes, classes).predict(attributes)

#print("\n" + 'linear' + ": Number of mislabeled points out of a total %d points : %d" % (attributes.shape[0],(classes != y_pred).sum()))


clf = svm.LinearSVC()

scores = cross_val_score(clf, attributes, classes, cv = crossValSize)

print("Actual Linear Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = clf.fit(attributes, classes).predict(attributes)

#print("\n" + "ACCUAL LINEAR" + ": Number of mislabeled points out of a total %d points : %d" % (attributes.shape[0],(classes != y_pred).sum()))

clf = AdaBoostClassifier(n_estimators=100)

scores = cross_val_score(clf, attributes, classes, cv = crossValSize)

print("AdaBoost Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = tree.DecisionTreeClassifier()

scores = cross_val_score(clf, attributes, classes, cv = crossValSize)

print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


