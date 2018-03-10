from nipype.interfaces.ants import N4BiasFieldCorrection
import sys
import os
import ast
# from skimage import io
import subprocess

# Shit to download:
# sudo pip3 install -U scikit-image
# easy_install SimpleITK
# sudo pip3 install -U nipype
# https://sourceforge.net/projects/advants/files/latest/download

#
# if len(sys.argv) < 2:
#     print("INPUT from ipython: run n4_bias_correction input_image dimension n_iterations(optional, form:[n_1,n_2,n_3,n_4]) output_image(optional)")
#     sys.exit(1)
#
# # if output_image is given
# if len(sys.argv) > 3:
#     n4 = N4BiasFieldCorrection(output_image=sys.argv[4])
# else:
#     n4 = N4BiasFieldCorrection()
#
# # dimension of input image, input image
# n4.inputs.dimension = int(sys.argv[2])
# n4.inputs.input_image = sys.argv[1]
#
# # if n_dinesions arg given
# if len(sys.argv) > 2:
#     n4.inputs.n_iterations = ast.literal_eval(sys.argv[3])
#
# n4.run()



# Test
# img = load_pickle_file('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/PickledData/train_data0.p')
# imgMod = img[:,:,:,1]
outPath = '/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Utilities/out.mha'
# io.imsave(outPath, imgMod,plugin='simpleitk')
# imgRead = io.imread(outPath,plugin='simpleitk')
# io.imshow(imgRead[:,:,80])
# plt.figure()
# plt.imshow(imgRead[:,:,80])
# plt.show()

path = outPath
n_dims = 3
n_iters = [20,20,10,5]
output_fn = '/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Utilities/outN4.mha'

subprocess.call('python N4_bias_correction.py ' + str(path) + ' ' + str(n_dims) + ' ' + str(n_iters)  + output_fn, shell=True)
#
# outFile = io.imread(out, plugin='simpleitk').astype(float)