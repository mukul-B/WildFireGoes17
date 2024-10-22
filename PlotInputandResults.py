import matplotlib.pyplot as plt
import numpy as np

from CommonFunctions import prepareDirectory
from EvaluationMetricsAndUtilities import denoralize, getth

plt.style.use('plot_style/wrf')


class ImagePlot:
    def __init__(self,unit,vmax,vmin,image_blocks,lable_blocks,binary=False):
        self.unit = unit
        self.vmin = vmin
        self.vmax = vmax
        self.binary = binary
        if(self.unit):
            self.image_blocks = denoralize(image_blocks,vmax,vmin)
        else:
            self.image_blocks = None
        self.lable_blocks = lable_blocks


def plot_from_ImagePlot(title,img_seq,path,shape=(128,128),colection=True):
    pl = path.split('/')
    filename = pl[-1].replace('.png','')
    c,r = len(img_seq) ,len(img_seq[0])
    if(colection):
        fig, axs = plt.subplots(c, r, constrained_layout=True, figsize=(12, 4*c))
        fig.suptitle(title)
    for col in range(c):
        for row in range(r):
            
            image_blocks=img_seq[col][row].image_blocks
            lable_blocks=img_seq[col][row].lable_blocks
            binary = img_seq[col][row].binary
            cb_unit=img_seq[col][row].unit
            vmin,vmax = img_seq[col][row].vmin, img_seq[col][row].vmax
            # vmin,vmax = 0 , 413
            if(colection):
                if(r == 1):
                    ax = axs[col]
                elif(c==1):
                    ax = axs[row]
                else:
                    ax = axs[col][row]
                ax.set_title(lable_blocks)
                

            if  image_blocks is not None:
                if(not colection):
                    fig = plt.figure()
                    fig.suptitle(title)
                    ax = fig.add_subplot()

                # X, Y = np.mgrid[0:1:128j, 0:1:128j]
                X, Y = np.mgrid[0:1:complex(str(shape[0]) + "j"), 0:1:complex(str(shape[1]) + "j")]
                
                # vmin = VIIRS_MIN_VAL if lable_blocks in [VIIRS_GROUND_TRUTH] else None
                # vmax =420

                # cb_unit = "Background | Fire Area     " if lable_blocks in [
                #     Prediction_JACCARD] else VIIRS_UNITS
                # if lable_blocks in [Prediction_JACCARD_LABEL,Prediction_Segmentation_label]:
                if binary:
                    sc = ax.pcolormesh(Y, -X, image_blocks, cmap=plt.get_cmap("gray_r", 2), vmin=vmin, vmax=vmax)
                else:
                    # sc = ax.pcolormesh(Y, -X, image_blocks, cmap=plt.get_cmap("gray_r"), vmin=vmin, vmax=vmax)
                    sc = ax.pcolormesh(Y, -X, image_blocks, cmap=plt.get_cmap("jet"), vmin=vmin, vmax=vmax)
                    cb = fig.colorbar(sc, pad=0.01,ax=ax)
                    cb.ax.tick_params(labelsize=11)
                    # cb.set_label(cb_unit, fontsize=12)
                    cb.set_label(cb_unit, fontsize=13)
                
                # plt.tick_params(left=False, right=False, labelleft=False,
                #                 labelbottom=False, bottom=False)
                # Hide X and Y axes label marks
                # ax.xaxis.set_tick_params(labelbottom=False)
                # ax.yaxis.set_tick_params(labelleft=False)
                
                # Hide X and Y axes tick marks
                ax.set_xticks([])
                ax.set_yticks([])
                if(not colection):
                    filenamecorrection = lable_blocks.replace('\n','_').replace(' ','_').replace('.','_').replace('(','_').replace(')','_').replace(':','_')
                    condition = ''
                    path ='/'.join(pl[:-1]) + f'/{condition}_{filename}_{filenamecorrection}.png'
                    # path = '/'.join(pl[:1]) + f"/allresults/{condition}/{filename}_{filenamecorrection}.png" 
                    # print(path)
                    # prepareDirectory(path)
                    plt.savefig(path,
                                bbox_inches='tight', dpi=600)
                    plt.show()
                    plt.close()
    if(colection):
        # path ="check.png"
        # path = '/'.join(pl[:1]) + f"allresults/{condition}/{filename}/{pl[:1]}.png" 
        # path = "cheko.png"
        # path = '/'.join(pl[:2]) + f"/{condition}/{filename[0]}_{output_iou}_{psnr_intersection_i}_{psnr_union_i}_{str(round(coverage_i, 4))}.png"
        print(filename)
        # path = '/'.join(pl[:-1]) + f"/{condition}"
        path = '/'.join(pl[:-1]) 
        # prepareDirectory(path)
        fpath = path + f'/{filename}.png'
        plt.rcParams['savefig.dpi'] = 600
        
        fig.savefig(fpath)
        # input()
        
        # plt.show()
        plt.close()


def plot_histogramandimage(image,path):
    # image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
    # histogram, bins = np.histogram(image, bins=256, range=(0, 256))
    ret2, th2, hist2, bins2, index_of_max_val2 = getth(image)

    

    # Create subplots with constrained layout
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 8))

    # Plot the image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Image")
    axs[0].axis('off')

    # Plot the histogram
    # axs[1].plot(hist2, color='black')
    axs[1].hist(bins2[1:-1], bins=bins2[1:], weights=hist2[1:], color='blue')
    axs[1].set_title("Histogram")
    axs[1].set_xlabel("Pixel Value")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlim(0, 255)
    # axs[1].set_ylim(0, 200)

    # show_hist([1], "Histogram", bins2, hist2, 1)
    

    plt.tight_layout()
    plt.savefig(path)
    plt.close()