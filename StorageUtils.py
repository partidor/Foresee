import cv2
import numpy as np
from image_match.goldberg import ImageSignature
import pickle
import glob
import CardUtils as cu
import os

gis = ImageSignature(12)

class Storage_struct:

	def __init__(self):
		self.hash_dict = dict()

	def import_images(self, path):
		for root, dirs, files in os.walk(path):
			print("walking this path: " + str(root))
			for filename in files:
				if filename.endswith(".jpg"):
					name = os.path.basename(filename)
					GeCardDetail, _ = os.path.splitext(name)
					GeCardDetail = GeCardDetail.split(' - ')
					mvid = GeCardDetail[0]
					CardName = GeCardDetail[1]
					sig = gis.generate_signature(os.path.join(root, filename))
					elm = cu.make_record(sig)
					self.insert_element(elm, [str(mvid), CardName])

	def insert_element(self, elm, cardData):
		signature = elm[0]

		for word in elm[1:]:
			self.hash_dict[word] = (signature, cardData)

	def check_element(self, elm):
		elm = gis.generate_signature(elm)
		elm = cu.make_record(elm)
		sig_to_check = elm[0]

		words_to_check = elm[1:]

		mvid = None
		cardName = None
		minScore = 100
		for word in words_to_check:
			if word in self.hash_dict:
				mtuple = self.hash_dict[word]
			else:
				continue
			testHash = mtuple[0]
			testScore = gis.normalized_distance(testHash, sig_to_check)
			if testScore < minScore and testScore < 0.42:
				minScore = testScore
				mvid = mtuple[1][0]
				cardName = mtuple[1][1]

		return mvid, cardName

	def import_dict(self, dictfile):

		p_in = open(dictfile, "rb")
		self.hash_dict = pickle.load(p_in)
		p_in.close()
		print("Prebuilt dict imported")
	
	def export_dict(self, dictfile):

		p_out = open(dictfile, "wb")
		pickle.dump(self.hash_dict, p_out)
		p_out.close()

	def rotate_image(image, angle):
		image_center = tuple(np.array(image.shape)/2)
		rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
		result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
		return result	
		

