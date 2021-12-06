rawdata = open('./testdatabase/transit_signal.tab','r')
output = open('./testdatabase/test_trans_sign.txt','w')
trainset = open('./testdatabase/trainset.txt','w')
output.write("UniprotID\tLength\tFeature\tStart\tEnd\tSequence\tNote\n")
line = rawdata.readline()
line = rawdata.readline()
while line:
    try:
        uniportid,name_tax,proname,length,signal,transit,sequence = line.rstrip('\n').split('\t')
        name,tax = name_tax.split('_')
        fse = (signal+transit).split('; ')[0]
        if fse:
            feature,start_end = fse.split(' ')
            start,end = start_end.split('..')
        else:
            feature,start,end = ['Negative','1',length]
        note = feature[0] * int(end) + 'N' * (int(length)-int(end))
        if feature[0] == 'S':
            weight = 32
        elif feature[0] == 'T':
            weight = 4
        else:
            weight = 1
        output.write('\t'.join([uniportid,length,feature,start,end,sequence,note])+'\n')
        trainset.write('\t'.join([sequence,note,str(weight)])+'\n')
        line = rawdata.readline()
    except:
        line = rawdata.readline()
rawdata.close()
output.close()
trainset.close()