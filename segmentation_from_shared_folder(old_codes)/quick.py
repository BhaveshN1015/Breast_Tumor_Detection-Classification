import nibabel as nib

p1 = nib.load(r"D:\Breast_Tumor_AI_Project\data\patients\train\1\P1.nii.gz")
gt = nib.load(r"D:\Breast_Tumor_AI_Project\data\patients\train\1\GT.nii.gz")

print("P1 shape  :", p1.shape)
print("P1 spacing:", p1.header.get_zooms())
print("GT shape  :", gt.shape)
print("GT spacing:", gt.header.get_zooms())
print("Affines match:", (p1.affine == gt.affine).all())