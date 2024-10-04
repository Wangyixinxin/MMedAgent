# Flare Writing
image_head = "****The path to image folder"
label_head = "****The path to label folder"
with open("****The path to save image.txt", 'w') as f1:
    for i in range(1, 51):
        path = image_head + "FLARE22_Tr_" + "0" * (4 - len(str(i))) + str(i) + "_0000.nii.gz"
        f1.write(path + '\n')

with open("****The path to save label.txt", 'w') as f2:
    for i in range(1, 51):
        path = label_head + "FLARE22_Tr_" + "0" * (4 - len(str(i))) + str(i) + ".nii.gz"
        f2.write(path + '\n')