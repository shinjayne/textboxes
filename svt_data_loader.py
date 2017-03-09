import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random

class SVT:
	def __init__(self, trainPath=None, testPath=None):
		trainList = None
		testList = None
		if trainPath:
			self.trainList = self.parseTree(trainPath)
		if testPath:
			self.testList = self.parseTree(testPath)

	def parseTree(self, path):
		dataset = []
		tree = ET.parse(path)
		root = tree.getroot()
		for image in root.findall('image'):
			name = image.find('imageName').text
			rectangles = []
			taggedRectangles = image.find('taggedRectangles')
			for rectangle in taggedRectangles.findall('taggedRectangle'):
				h = int(rectangle.get('height'))
				w = int(rectangle.get('width'))
				x = int(rectangle.get('x'))
				y = int(rectangle.get('y'))
				rectangles.append(([x,y,w,h], 1))
			dataset.append((name, rectangles))
		return dataset

	def nextBatch(self, batches, dataset='train'):
		imgH = 300
		imgW = 300
		if dataset == 'train':
			datalist = self.trainList
		if dataset == 'test':
			datalist = self.testList
		randomIndex = random.sample(range(len(datalist)), batches)
		images = []
		anns = [datalist[x][1] for x in randomIndex]
		for index in randomIndex:
			fileName = './svt1/' + datalist[index][0]
			img = cv2.imread(fileName, cv2.IMREAD_COLOR)
			resized = cv2.resize(img, (imgW, imgH))
			resized = np.multiply(resized, 1.0/255.0)
			images.append(resized)
		images = np.asarray(images)
		return (images, anns)

if __name__ == '__main__':
	loader = SVT('./svt1/train.xml', './svt1/test.xml')
	train_img, train_anns = loader.nextBatch(5,'test')
	for img, anns in zip(train_img, train_anns):
		cv2.imshow('output', img)
		cv2.waitKey(0)

