from __future__ import division
import cv2
import os
import glob
import csv
import numpy as np
# from baseline import sample_idx

sample_idx = 16
train_path = 'dataset/train/'
test_path = 'dataset/test/'
folders = os.listdir(test_path)
fileName = 'aicon_train_pix.csv'
idxName = 'aicon_train_pix_idx.csv'

if os.path.isfile(fileName):
    f = open(fileName, 'r')
    read = csv.reader(f)
    fw = open(idxName, 'a', newline='')
    write = csv.writer(fw)

    for line in read:
        line[1] = line[1].replace('[', '').replace(']', '')
        line[2] = line[2].replace('[', '').replace(']', '')
        line1 = line[1].split(',')
        line2 = line[2].split(',')
        sumPix = [float(x) for x in line1]
        avgPix = [float(x) for x in line2]
        sumStart = sumPix[1:(len(sumPix)//4)]
        avgStart = avgPix[1:(len(avgPix)//4)]
        sumStart = np.array(sumStart)
        avgStart = np.array(avgStart)
        sumEnd = sumPix[(len(sumPix)//2):]
        avgEnd = avgPix[(len(avgPix)//2):]
        sumEnd = np.array(sumEnd)
        avgEnd = np.array(avgEnd)

        maxIdx = sumStart.argmax()
        minIdx = sumEnd.argmin() + len(sumPix)//2
        maxAvgIdx = avgStart.argmax()
        minAvgIdx = avgEnd.argmin()

        if maxIdx - sample_idx//2 < 0 :
            start_idx = 0
        else :
            start_idx = maxIdx - sample_idx//2

        write.writerow([line[0], start_idx, maxIdx, minIdx])

    f.close()

else:

    f = open(fileName, 'a', newline='')
    write = csv.writer(f)

    for folder in folders :
        fol_path = train_path + folder + '/'
        subfolders = os.listdir(fol_path)

        for folder in subfolders:
            subfol_path = fol_path + folder + '/'
            frames = os.listdir(subfol_path)

            pixSum = []
            pixAvg = []

            for idx, frame in enumerate(frames):
                if idx > len(frames) - 2:
                    continue

                img_path = os.path.join(subfol_path, frame)
                img_path_2 = os.path.join(subfol_path, frames[idx + 1])
                img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
                cv2.resize(img1, (112, 112))
                cv2.resize(img2, (112, 112))
                subImg = img2 - img1
                sumImg = cv2.sumElems(subImg)
                avgImg = cv2.mean(subImg)
                pixSum.append(max(sumImg))
                pixAvg.append(max(avgImg))

            write.writerow([folder, pixSum, pixAvg])

    f.close()

    f = open(fileName, 'r')
    read = csv.reader(f)
    fw = open(idxName, 'a', newline='')
    write = csv.writer(fw)

    for line in read:
        line[1] = line[1].replace('[', '').replace(']', '')
        line[2] = line[2].replace('[', '').replace(']', '')
        line1 = line[1].split(',')
        line2 = line[2].split(',')
        sumPix = [float(x) for x in line1]
        avgPix = [float(x) for x in line2]
        sumStart = sumPix[1:(len(sumPix) // 4)]
        avgStart = avgPix[1:(len(avgPix) // 4)]
        sumStart = np.array(sumStart)
        avgStart = np.array(avgStart)
        sumEnd = sumPix[(len(sumPix) // 2):]
        avgEnd = avgPix[(len(avgPix) // 2):]
        sumEnd = np.array(sumEnd)
        avgEnd = np.array(avgEnd)

        maxIdx = sumStart.argmax()
        minIdx = sumEnd.argmin() + len(sumPix) // 2
        maxAvgIdx = avgStart.argmax()
        minIdx = avgEnd.argmin()

        if maxIdx - sample_idx // 2 < 0:
            start_idx = 0
        else:
            start_idx = maxIdx - sample_idx // 2

        write.writerow([line[0], start_idx, maxIdx, minIdx])

f.close()
fw.close()
