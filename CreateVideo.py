import imageio
import glob

from GlobalValues import RealTimeIncoming_results, videos

image_list = []
img_src = RealTimeIncoming_results
dir = {}
print('\nCompiling image list...', end='', flush=True)
count = 0
for img in sorted(glob.glob(img_src + '/*.png')):
    try:
        im = imageio.imread(img)
        dir[im.size] = dir.get(im.size,-1) + 1
        # im = cv2.resize(im, (900, 900)) #manually resize if needed
        if(im.size==5248512):
            image_list.append(im)
    except:
        count += 1
        print(img)
        continue

print(dir,count)
print('Creating mp4 video...', end='', flush=True)
imageio.mimwrite(videos + '/GOES_superResolution.mp4', image_list, fps=36)
print('Done.\n')