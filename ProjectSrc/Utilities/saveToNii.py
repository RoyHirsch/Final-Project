import numpy as np
from Utilities.loadDataNii import *
import nibabel as nib


images, labels, filenames = get_data_and_labels_from_folder()
print('images shape')
print(np.shape(images))
print('labels shape')
print(np.shape(labels))
print(filenames)
modalities =['_T1', '_T2', '_T1g', '_FLAIR']

dest = '/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/NiiData/'
for ind in range(len(images)):
	image = images[ind]
	filteredFilename = filenames[ind][0].split('_')[3]
	print('saving image: {}'.format(filteredFilename))
	for mod in range(4):
		new_image = nib.Nifti1Image(image[:,:,:,mod], affine=np.eye(4))
		nib.save(new_image, dest + filteredFilename + modalities[mod] + '.nii')
		print('saved image named: {} as nii image'.format(filteredFilename + modalities[mod]))
