import cv2
import imutils
import numpy as np
import time
import CardUtils as cu
import StorageUtils as su
import imagehash
import urllib
from image_match.goldberg import ImageSignature
import collections
from collections import defaultdict
from skimage.measure import compare_ssim

mvidCounter = dict()
mvidCounter = defaultdict(lambda: 0, mvidCounter)
CurrentCardName = None
sleeved = True
TIMER = 0
convert = 0
maxConvert = 20
minConvert = maxConvert // 2
mc = -5
xc = 5

BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200

#gis = ImageSignature(11, (0, 100))
gis = ImageSignature(12)
storage = su.Storage_struct()

#storage.import_images('C:\\Users\\J\\Desktop\\cardreader2')

storage.import_dict("C:\\Users\\J\\Desktop\\cardreader2\\dict.pickle")

BKG_THRESH = 50
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 2500

cap = cv2.VideoCapture(0)

ret, first = cap.read()
first = cv2.flip(first, -1)
fgray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)	

ham = []

TESTRECORD = None
#mvidPrev = collections.deque(maxlen=10)
EMPTY = True
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

while True:


	ret, frame = cap.read()
	frame = cv2.flip(frame, -1)
	image = frame.copy() #copy frame so that we don't get funky contour problems when drawing contours directly onto the frame.

    

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray = cv2.bilateralFilter(gray, 11, 17, 17) 
	#gray = cv2.GaussianBlur(gray,(1,1),1000)

	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)


	cv2.imshow("Edge map", edges)
	#cv2.imshow("Web Cam", frame)

	#find contours in the edged image, keep only the largest
	# ones, and initialize our screen contour
	_, cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
	box = []
	newCard = None

	# loop over our contours
	for c in cnts:
		# approximate the contour
		size = cv2.contourArea(c)
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.1 * peri, True)

		# if our approximated contour has four points, then
		# we can assume that we have found our card
		if len(approx) == 4 and (size < CARD_MAX_AREA) and (size > CARD_MIN_AREA):
			box = approx
			newCard = cu.process_card(box, image, convert)
			break
	#convert = ((convert + 1) % maxConvert) - minConvert
	#convert = -10
	convert -= 1
	if convert < -9:
		convert = 2

	if type(box) != 'NoneType' and len(box) != 0:
		cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
		EMPTY = False
	elif len(box) != 4:
		TIMER += 1
		if TIMER < 100000:
			EMPTY = True
			TIMER = 0
			if CurrentCardName != None:
			#if len(mvidPrev) != 0:
				#mvidPrev.clear()
				CurrentCardName = None
				print("card queue clear")

	cv2.imshow("Webcam Feed w/ Contour", image)

	if newCard != None and CurrentCardName == None:

		if not sleeved:
			im = newCard.warp.copy()
			cv2.imshow("Card", im)
		else:
			im = newCard.sleeved.copy()
			cv2.imshow("Card", im)

		im = clahe.apply(im)


		mvid, cname = storage.check_element(im)
		fim = cv2.flip(im, -1)
		mvidType = "0 Degrees"
		if mvid is None:
			mvid, cname = storage.check_element(fim)
			mvidType = "flip"
		if mvid is None:
			mvid, cname = storage.check_element(imutils.rotate(im, 1))
			if mvid is None:
				mvid, cname = storage.check_element(imutils.rotate(fim, 1))
			mvidType = "1 Degrees"
		if mvid is None:
			mvid, cname = storage.check_element(imutils.rotate(im, 1.5))
			if mvid is None:
				mvid, cname = storage.check_element(imutils.rotate(fim, 1.5))
			mvidType = "1.5 Degrees"
		if mvid is None:
			mvid, cname = storage.check_element(imutils.rotate(im, 2))
			if mvid is None:
				mvid, cname = storage.check_element(imutils.rotate(fim, 2))
			mvidType = "2 Degrees"
		if mvid is None:
			mvid, cname = storage.check_element(imutils.rotate(im, -1))
			if mvid is None:
				mvid, cname = storage.check_element(imutils.rotate(fim, -1))
			mvidType = "-1 Degrees"
		if mvid is None:
			mvid, cname = storage.check_element(imutils.rotate(im, -1.5))
			if mvid is None:
				mvid, cname = storage.check_element(imutils.rotate(fim, -1.5))
			mvidType = "-1.5 Degrees"
		if mvid is None:
			mvid, cname = storage.check_element(imutils.rotate(im, -2))
			if mvid is None:
				mvid, cname = storage.check_element(imutils.rotate(fim, -2))
			mvidType = "-2 Degrees"
		if mvid is None:
			mvid, cname = storage.check_element(imutils.rotate(im, 0.5))
			if mvid is None:
				mvid, cname = storage.check_element(imutils.rotate(fim, 0.5))
			mvidType = "0.5 Degrees"
		if mvid is None:
			mvid, cname = storage.check_element(imutils.rotate(im, -0.5))
			if mvid is None:
				mvid, cname = storage.check_element(imutils.rotate(fim, -0.5))
			mvidType = "-0.5 Degrees"
		#Change this if to deal with new cname value passed from check_element
		if (mvid is not None) and (cname != CurrentCardName) and not EMPTY:
			url = "http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=" + mvid + "&type=card"
			url_response = urllib.request.urlopen(url)
			img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
			if len(img_array) == 0:
				print("Image Error on " + mvid + ", probably a Token or special Promo")
				#mvidPrev.append(mvid)
				CurrentCardName = cname
			else:
				print(len(img_array))
				img = cv2.imdecode(img_array, -1)
				cv2.imshow('Scan', img)
				print(mvid)
				print(mvidType)
				#mvidPrev.append(mvid)
				CurrentCardName = cname
				print(cname)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()

cv2.destroyAllWindows()

