#TODO: to be merged with PlotInputandResult.py
import matplotlib.pyplot as plt
import numpy as np



plt.style.use('plot_style/wrf')

class ImagePlot2:
    def __init__(self,to_plot, lables,units,color_map):
        self.to_plot, self.lables, self.units, self.color_map = to_plot, lables,units,color_map
        self.vmin=None
        self.vmax=None

def Plot_list( title, imges_plots, shape, save_path=None):

    X, Y = np.mgrid[0:1:complex(str(shape[0]) + "j"), 0:1:complex(str(shape[1]) + "j")]
    n = len(imges_plots)
    fig, ax = plt.subplots(1, n, constrained_layout=True, figsize=(4*n, 4))
    fig.suptitle(title)
    
    for k in range(n):
        curr_image = imges_plots[k]
        to_plot, lables,units,color_map ,vmin,vmax = curr_image.to_plot, curr_image.lables, curr_image.units, curr_image.color_map , curr_image.vmin, curr_image.vmax 
        # vminc, vmaxc = (0 , 3500) if(color_map[k] == 'terrain') else (vmin, vmax)
        unit = units
        image_blocks = to_plot
        lable_blocks = lables
        cmap=color_map
        
        curr_img = ax[k] if n > 1 else ax
        p = curr_img.pcolormesh(Y, -X, image_blocks, cmap=cmap, vmin=vmin, vmax=vmax)
        curr_img.tick_params(left=False, right=False, labelleft=False,
                          labelbottom=False, bottom=False)
        curr_img.text(0.5, -0.1, lable_blocks, transform=curr_img.transAxes, ha='center', fontsize=12)

        cb = fig.colorbar(p, pad=0.01)
        cb.ax.tick_params(labelsize=11)
        cb.set_label(unit, fontsize=12)
    plt.rcParams['savefig.dpi'] = 600
    if (save_path):
        fig.savefig(save_path)
        plt.close()
    plt.show()


def Plot_individual_list( title, imges_plots, shape, save_path=None):

    X, Y = np.mgrid[0:1:complex(str(shape[0]) + "j"), 0:1:complex(str(shape[1]) + "j")]
    n = len(imges_plots)
    # fig, ax = plt.subplots(1, n, constrained_layout=True, figsize=(4*n, 4))
    # fig.suptitle(title)
    
    for k in range(n):
        curr_image = imges_plots[k]
        to_plot, lables,units,color_map ,vmin,vmax = curr_image.to_plot, curr_image.lables, curr_image.units, curr_image.color_map , curr_image.vmin, curr_image.vmax 
        # vminc, vmaxc = (0 , 3500) if(color_map[k] == 'terrain') else (vmin, vmax)
        unit = units
        image_blocks = to_plot
        lable_blocks = lables
        cmap=color_map
        
        fig = plt.figure()
        ax = fig.add_subplot()
        curr_img =  ax
        p = curr_img.pcolormesh(Y, -X, image_blocks, cmap=cmap, vmin=vmin, vmax=vmax)
        curr_img.tick_params(left=False, right=False, labelleft=False,
                          labelbottom=False, bottom=False)
        curr_img.text(0.5, -0.1, lable_blocks, transform=curr_img.transAxes, ha='center', fontsize=12)

        cb = fig.colorbar(p, pad=0.01)
        cb.ax.tick_params(labelsize=11)
        cb.set_label(unit, fontsize=12)
        plt.rcParams['savefig.dpi'] = 600
        if (save_path):
            save_pathm = save_path.replace('.png',f'_{lables}.png')
            fig.savefig(save_pathm)
            plt.close()
        else:
            plt.show()

def plot_individual_images(X, Y, compare_dir, g_file, gd, vd):
    ar = [vd, gd, gd - vd]
    if (g_file == 'reference_data/Kincade/GOES/ABI-L1b-RadC/tif/GOES-2019-10-27_949.tif'):
        print(g_file, compare_dir)
        for k in range(3):
            fig2 = plt.figure()
            ax = fig2.add_subplot()
            a = ax.pcolormesh(Y, -X, ar[k], cmap="jet", vmin=200, vmax=420)
            cb = fig2.colorbar(a, pad=0.01)
            cb.ax.tick_params(labelsize=11)
            cb.set_label('Radiance (K)', fontsize=12)
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            # plt.show()
            plt.savefig(f'{compare_dir}/data_preprocessing{k}.png', bbox_inches='tight', dpi=600)
            plt.close()
            
