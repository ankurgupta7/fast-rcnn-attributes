ft = open('train.txt', 'w')
for i in range(1,4011):
    ft.write('{:05d}\n'.format(i))

ft.close()
