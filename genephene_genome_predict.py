#Script to predict phenotypic functions from gene orthologs

#load required modules

import pandas as pd
import numpy as np
import argparse
import pickle

#plotting
#from matplotlib import pyplot as plt
#from matplotlib import style
#style.use('ggplot')

#constants and options

clades = ['Kingdom','Phylum','Class','Order','Family','Genus','Species']

import warnings
warnings.filterwarnings('ignore')

#define functions 

def load_classifiers(orth_type):
	'''Load the list of classifiers used to make predictions from the pickle file.'''
	
	if orth_type=='Pfam':
		clf_dict = pickle.load(open('binaries/pfam_classifiers.pickle','rb'))
		orth_list = list(pd.read_csv('datafiles/pfam_ortholog_list.txt', header=None)[0])
	elif orth_type=='KO':
		clf_dict = pickle.load(open('binaries/ko_classifiers.pickle','rb'))
		orth_list = list(pd.read_csv('datafiles/ko_ortholog_list.txt', header=None)[0])
	else:
		raise ValueError('Ortholog type not recognized. Must be Pfam or KO.')

	return clf_dict, orth_list


def load_input_ortholog_file(fpath, training_ortholog_list):
	'''Load the user-supplied data file containing MAG/genome IDs and their ortholog copy numbers.'''

	orth_table = pd.read_csv(fpath)

	try:
		orth_table = orth_table.set_index('genome_ID')
	except KeyError:
		print("Error: Input ortholog table must have a field called 'genome_ID' with genome identifiers.")
		raise


	#check that there is some overlap between the supplied orthologs and the training ones
	user_orths = orth_table.columns
	try:
		assert(set(user_orths) & set(training_ortholog_list))
	except:
		print("Error: no overlap between training and supplied orthologs. Are you using the correct ortholog set (KO/Pfam)?")
		raise

	#remove any orthologs not in the training data
	orth_table = orth_table[training_ortholog_list]

	print("Loaded %i genomes. %i orthologs can be used for prediction."%orth_table.shape)
	return orth_table

def predict_functions(user_orth_table, clf_dict):

	predictions = pd.DataFrame()

	for func in clf_dict:
	    clf = clf_dict[func]
	    predictions[func] = clf.predict(user_orth_table)
	    
	predictions.index = user_orth_table.index

	print('Predicted %i total functions.'%predictions.sum().sum())

	return predictions


if __name__=='__main__':

	#command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_table', required=True, help="Filepath to a CSV file containing the list of genomes, where the columns are gene orthologs and the rows are genomes, with the entries being gene copy numbers. Must have a header row of the form 'genome_ID,orth1,orth2,orth3'...")
	parser.add_argument('-o', '--output_file', required=True, help="Output filepath for the prediction table. CSV file with format 'genome_ID,func1,func2,...' where the entries are 0/1 for absence/presence of the function.")
	parser.add_argument('-g', '--gene_ortholog_type', required=False, default='KO', help="Gene ortholog/family database to use. Must be either 'KO' or 'Pfam'")

	args = parser.parse_args()

	#load application data
	clf_dict, training_ortholog_list = load_classifiers(args.gene_ortholog_type)

	#load user data
	orth_table = load_input_ortholog_file(args.input_table, training_ortholog_list)

	print()

	#make the predictions
	print('Predicting functions...')
	predictions = predict_functions(orth_table, clf_dict)
	predictions.to_csv(args.output_file)








	