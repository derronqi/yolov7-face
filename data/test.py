import os
import glob
import numpy as np

txtlist = glob.glob('widerface/val/*.txt')
for txt in txtlist:
	dst = txt.replace('val', 'tmp')
	fw = open(dst, 'w')
	with open(txt, 'r') as f:
		lines = f.readlines()
		for line in lines:
			data = np.array(line.strip().split(),dtype=np.float32)
			print(line)
			if len(np.where(data < 0)[0]) == 10:
				label = '0 {:.4f} {:.4f} {:.4f} {:.4f} 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000'.format(data[1],data[2],data[3],data[4])
			else:
				label = '0 {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} 2.0000 {:.4f} {:.4f} 2.0000 {:.4f} {:.4f} 2.0000 {:.4f} {:.4f} 2.0000 {:.4f} {:.4f} 2.0000'.format(data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[13],data[14])
			fw.write(label + '\n')
	fw.close()
			

