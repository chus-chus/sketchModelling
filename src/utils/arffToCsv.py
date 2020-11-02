#########################################
# Project   : ARFF to CSV converter     #
# Created   : 10/01/17 11:08:06         #
# Author    : haloboy777                #
# Licence   : MIT                       #
#########################################


# Importing library
import os


# Function for converting arff list to csv list
def to_csv(content):
    data = False
    header = ""
    newContent = []
    for line in content:
        if not data:
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute") + 1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent


def arff_to_csv(filename):
    with open(filename, "r") as inFile:
        fileContent = inFile.readlines()
        name, ext = os.path.splitext(inFile.name)
        new = to_csv(fileContent)
        with open(name + ".csv", "w") as outFile:
            outFile.writelines(new)


if __name__ == "__main__":

    # Getting all the arff files from the current directory
    files = [arff for arff in os.listdir('.') if arff.endswith(".arff")]

    # Main loop for reading and writing files
    for file in files:
        arff_to_csv(file)
