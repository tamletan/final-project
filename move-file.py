from pathlib import Path
import os

dst_path = r".\dataset\data-extract\e"

if not os.path.isdir(dst_path):
	os.mkdir(dst_path)

f = open(r".\dataset\data-extract\error-file.txt", "r")
for x in f:
	x = x.replace('\n','')
	file = x.split('/')[-1]
	dst = os.path.join(dst_path,file)
	print(dst)
	Path(x).rename(dst)

f.close()