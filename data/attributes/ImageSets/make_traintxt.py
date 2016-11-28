f = open('train.txt', 'w')
for i in range(1,4011):
    f.write('{:05d}\n'.format(i))

f.close()
