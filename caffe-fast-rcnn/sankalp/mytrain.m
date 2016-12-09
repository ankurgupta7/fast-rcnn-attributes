model = 'age.prototxt';
weights = 'dex_chalearn_iccv2015.caffemodel';
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(model, weights, 'test');

image = imread('1.jpg');
res = net.forward({image});
prob = res{1}
print prob
