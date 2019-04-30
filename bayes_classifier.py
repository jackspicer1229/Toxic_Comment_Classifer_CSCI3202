from collections import Counter
import string, math
import pandas as pd


#Develops a unigram model given a .csv file of toxic/non-toxic Wikipedia comments
def count_words(file_name):
	toxic_data_frame = pd.read_csv(file_name)
	neutral_word_count = []
	toxic_word_count = []

	#Reads in data from training set and creates a corpus of words from all comments labeled negative and all comments unlabeled
	for index, row in toxic_data_frame.iterrows():
		if(row['toxic'] == 1 or row['severe_toxic'] == 1 or row['obscene'] == 1 or row['threat'] == 1 or row['insult'] == 1 or row['identity_hate'] == 1):
			for word in row['comment_text'].split():
				word = word.lower()
				word = word.translate(str.maketrans('','',string.punctuation))
				toxic_word_count.append(word)
		else:
			for word in row['comment_text'].split():
				word = word.lower()
				word = word.translate(str.maketrans('','',string.punctuation))
				neutral_word_count.append(word)

	return Counter(neutral_word_count), Counter(toxic_word_count)


#Returns either toxic or neutral based off of a Naive Bayes model and an input sentence
def compute_class_prob(sentence, neutral_counts, toxic_counts):
	neutral_vocab_size = len(neutral_counts)
	toxic_vocab_size = len(toxic_counts)
	neutral_prob = 0
	toxic_prob = 0


	#Naive Bayes
	for word in sentence:
		if word in neutral_counts:
			neutral_prob = neutral_prob + math.log((neutral_counts[word]+1)/(sum(neutral_counts.values())+neutral_vocab_size))
		else:
			neutral_prob = neutral_prob + math.log((1)/(sum(neutral_counts.values())+neutral_vocab_size))


		if word in toxic_counts:
			toxic_prob = toxic_prob + math.log((toxic_counts[word]+1)/(sum(toxic_counts.values())+toxic_vocab_size))
		else:
			toxic_prob = toxic_prob + math.log((1)/(sum(toxic_counts.values())+toxic_vocab_size))

	#Choose the more likely of the two classes
	if toxic_prob > neutral_prob:
		return "toxic"
	else:
		return "neutral"



def main():
	#Create word counts and initialize variables
	neutral_word_counts, toxic_word_counts = count_words("toxic_comment_data/train.csv")
	output = []
	id_predictions = []
	answers = []
	correct_answers = 0
	non_null_answers = 0
	num_test_texts = 20000

	#Read in the testing data and correct answers
	test_set = "toxic_comment_data/test.csv"
	test_data_frame = pd.read_csv(test_set)
	test_labels = "toxic_comment_data/test_labels.csv"
	test_labels_data_frame = pd.read_csv(test_labels)

	for index, row in test_data_frame.head(num_test_texts).iterrows():
		#Normalize input
		sentence = row['comment_text']
		sentence = sentence.lower()
		sentence = sentence.translate(str.maketrans('','',string.punctuation))
		sentence_list = sentence.split()

		#Classify current sentence using NB
		predicted_class = compute_class_prob(sentence_list, neutral_word_counts, toxic_word_counts)

		print(predicted_class)
		output.append(row['id'] + "\t" + predicted_class)
		id_predictions.append(predicted_class)


	#Check accuracy of our model on the testing data
	for index, row in test_labels_data_frame.head(num_test_texts).iterrows():
		answer = "null"
		if(row['toxic'] == 1 or row['severe_toxic'] == 1 or row['obscene'] == 1 or row['threat'] == 1 or row['insult'] == 1 or row['identity_hate'] == 1):
			if(id_predictions[index] == "toxic"):
				answer = "correct"
				correct_answers += 1
				non_null_answers += 1
			else:
				answer = "incorrect"
				non_null_answers += 1
		elif(row['toxic'] == 0 and row['severe_toxic'] == 0 and row['obscene'] == 0 and row['threat'] == 0 and row['insult'] == 0 and row['identity_hate'] == 0):
			if(id_predictions[index] == "neutral"):
				answer = "correct"
				correct_answers += 1
				non_null_answers += 1
			else:
				answer = "incorrect"
				non_null_answers += 1
		print(answer)
		output[index] = output[index] + "\t" + answer + "\n"


	with open('test_set_classification.txt', 'w') as f:
		for item in output:
			f.write(item)

	print("Number of correct answers: " + str(correct_answers))
	print("Percentage of correct answers: " + str(correct_answers/non_null_answers))
	

if __name__ == '__main__':
	main()