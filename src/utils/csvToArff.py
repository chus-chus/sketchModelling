
import os
import argparse
import numpy as np

# @author Jesus Antonanzas

# ASSUMES FILE CONTAINS HEADER
# Only supporting numeric and string types for now.
# Supports nominal, numeric and string target
# Not supported indexed csvs

# relationName will only be correctly extracted from path if separators are '/'


def to_arff(content, relationName, targetType, targetValues):
	attrTypes = [[attrName.replace('\n', ''), None] for attrName in content[0].split(',')]
	# remove header from csv
	content.pop(0)
	# check attribute types
	for index, attrValue in enumerate(content[0].split(',')):
		try:
			float(attrValue.replace('\n', ''))
			attrTypes[index][1] = 'NUMERIC'
		except ValueError:
			attrTypes[index][1] = 'STRING'
	content.insert(0, '@DATA\n')
	# write attributes, assumes target is the last one
	for index, attr in enumerate(reversed(attrTypes)):
		attrName = attr[0]
		attrType = attr[1]
		if index == 0:
			# target
			if targetType == 'ORDINAL':
				attrType = str({value for value in targetValues}).replace(' ', '').replace("'", '')
		content.insert(0, '@ATTRIBUTE ' + attrName + ' ' + attrType + '\n')
	content.insert(0, '@RELATION ' + relationName + '\n')
	return content


def csv_to_arff(filename, targetType='NOMINAL', targetValues=None, relationName=None):
	# targetType = {'ORDINAL', 'NOMINAL'}
	# targetValues (if targetType not None) = array-like('val1', 'val2')
	with open(filename, "r") as inFile:
		fileContent = inFile.readlines()
		namePath, ext = os.path.splitext(inFile.name)
		name = namePath.split('/')[-1]
		if relationName is None:
			relationName = name
		newFile = to_arff(fileContent, relationName, targetType, targetValues)
		with open(namePath + '.arff', 'w') as outFile:
			outFile.writelines(newFile)


def pd_to_arff(df, name, path, targetType='NOMINAL', targetValues=None):
	content = df.values
	with open(path + name + '.arff', "w") as outFile:
		# insert attributes
		attrTypes = [[attrName, None] for attrName in df.columns]
		for index, attrValue in enumerate(content[0, :]):
			try:
				float(attrValue)
				attrTypes[index][1] = 'NUMERIC'
			except ValueError:
				attrTypes[index][1] = 'STRING'

		outFile.write('@RELATION ' + name + '\n')
		for index, attr in enumerate(attrTypes):
			attrName = attr[0]
			attrType = attr[1]
			if index == len(attrTypes) - 1:
				# target
				if targetType == 'ORDINAL':
					attrType = str({value for value in targetValues}).replace(' ', '').replace("'", '')
			outFile.write('@ATTRIBUTE ' + attrName + ' ' + attrType + '\n')

		outFile.write('@DATA\n')
		for row in content:
			outFile.write(str(row).replace('  ', ',').replace(' ', ',').replace('[', '').replace(']', '').replace("'", '').replace("\n", '') + '\n')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')

	parser.add_argument('--filename', metavar='f', type=str, default=None)

	args = parser.parse_args()

	if args.filename is None:
		# Getting all the csv files from the current directory
		files = [csv for csv in os.listdir('.') if csv.endswith(".csv")]

		for file in files:
			csv_to_arff(file)

	else:
		csv_to_arff(args.filename, targetType='ORDINAL', targetValues=[0, 1])


