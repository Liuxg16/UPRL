import csv 
import utils
import random
file_name = 'data/quora/train.csv'
trainpath = 'data/train.txt'
testpath = 'data/test.txt'
referpath = 'data/refer.txt'
train_object = open(trainpath, 'w')
test_object = open(testpath, 'w')
refer_object = open(referpath, 'w')
testnum = 0
with open(file_name) as f:
    csv_reader = csv.reader(f, delimiter=',')
    title = True
    for row in csv_reader:
        
        ques1 = utils.clarify(row[3]).lower()
        ques2 = utils.clarify(row[4]).lower()

        if row[5]== '1' and testnum<30000 and len(ques1.split())<15 and random.random()>0.5:
            test_object.write(ques1)
            test_object.write('\n')
            refer_object.write(ques2)
            refer_object.write('\n')
            testnum+=1
        else:
            train_object.write(ques1)
            train_object.write('\n')
            train_object.write(ques2)
            train_object.write('\n')
test_object.close( )
refer_object.close( )

file_name = 'data/quora/test.csv'
with open(file_name) as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        ques1 = utils.clarify(row[3])
        ques2 = utils.clarify(row[4])

        train_object.write(ques1)
        train_object.write('\n')
        train_object.write(ques2)
        train_object.write('\n')

train_object.close( )
