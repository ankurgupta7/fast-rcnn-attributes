function VOCopts = get_voc_opts(path)

run /home/agupta82/fast-rcnn/data/attributes/VOCcode/VOCinit.m
%tmp = pwd
%cd(path)
%try
%  addpath('VOCcode');
%   %cd(path);
%   %addpath('VOCcode');
%  /home/agupta82/fast-rcnn/data/attributes/VOCcode/VOCinit.m;
%catch
%  rmpath('VOCcode');
%  cd(tmp);
%  error(sprintf('VOCcode directory not found under %s', path));
%end
%rmpath('VOCcode');
%cd(tmp);
%