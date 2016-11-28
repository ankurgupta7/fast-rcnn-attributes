f = open('labels.txt')
while True:
    l = f.readline()
    if len(l) == 0:
        break
    words = l.split()
    labels = ['is_male',  'has_long_hair',  'has_glasses',  'has_hat',  'has_t-shirt',  'has_long_sleeves',  'has_shorts',  'has_jeans',  'has_long_pants']
    output_str = ''
    for word in words[1:5]:
        output_str += word + ' '
    for i in range(words.__len__()):
        if (words[i] == '1'):
            output_str += labels[i - 5] + ' '
    output_str = output_str[0:-1]
    output_str = output_str.replace('NaN NaN NaN NaN', '0.0 0.0 20.0 20.0')
    f1 = open('../Annotations/'+ words[0].replace('jpg','txt'), 'w')
    f1.write(output_str)
    f1.close()
    #raw_input('Proceed?')
