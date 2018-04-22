data_dir = 'data/test'
import glob
import os
paths = glob.glob(data_dir + '/*/*')
gifs = glob.glob(data_dir + '/*/*.gif')
print('All -> jpg')
for path in paths:
    resol = path.split('.')[-1]
    if resol != 'jpg':
        os.system("convert " + path + ' ' + path + '.jpg')
        os.remove(path)
        #print('Image %s removed' % path)
        
new_paths = glob.glob(data_dir + '/*/*')
for path in new_paths:
    if path.split('-')[0] in gifs:
        if path[-6:] != '-0.jpg':
            os.remove(path)
            #print(path)
            
new_paths = glob.glob(data_dir + '/*/*')
for path in new_paths:
    if path[-6:] == '-0.jpg':
        os.rename(path, path[:-6]+'.jpg')

print('Done!')