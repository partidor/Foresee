import cv2
import numpy as np

wMult = 63
hMult = 88

class Magic_card:
	"""Structure to store information about query cards in the camera image."""

	def __init__(self):
		self.contour = [] # Contour of card
		self.width, self.height = 0, 0 # Width and height of card
		self.corner_pts = [] # Corner points of card
		self.center = [] # Center point of card
		self.warp = [] # 200x300, flattened, grayed, blurred image
		self.art = []
		self.sleeved = []

def flattener(image, pts, w, h, c):
	"""Flattens an image of a card into a top-down 200x300 perspective.
	Returns the flattened, re-sized, grayed image.
	See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
	temp_rect = np.zeros((4,2), dtype = "float32")
	
	s = np.sum(pts, axis = 2)

	tl = pts[np.argmin(s)]
	br = pts[np.argmax(s)]

	diff = np.diff(pts, axis = -1)
	tr = pts[np.argmin(diff)]
	bl = pts[np.argmax(diff)]

	convert = c


	tl[0][0] = tl[0][0] - convert
	tl[0][1] = tl[0][1] - convert
	tr[0][0] = tr[0][0] + convert
	tr[0][1] = tr[0][1] - convert
	bl[0][0] = bl[0][0] - convert
	bl[0][1] = bl[0][1] + convert
	br[0][0] = br[0][0] + convert
	br[0][1] = br[0][1] + convert

	# Need to create an array listing points in order of
	# [top left, top right, bottom right, bottom left]
	# before doing the perspective transform

	if w <= 0.8*h: # If card is vertically oriented
		temp_rect[0] = tl
		temp_rect[1] = tr
		temp_rect[2] = br
		temp_rect[3] = bl

	if w >= 1.2*h: # If card is horizontally oriented
		temp_rect[0] = bl
		temp_rect[1] = tl
		temp_rect[2] = tr
		temp_rect[3] = br

	# If the card is 'diamond' oriented, a different algorithm
	# has to be used to identify which point is top left, top right
	# bottom left, and bottom right.
	
	if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
		# If furthest left point is higher than furthest right point,
		# card is tilted to the left.
		if pts[1][0][1] <= pts[3][0][1]:
			# If card is titled to the left, approxPolyDP returns points
			# in this order: top right, top left, bottom left, bottom right
			temp_rect[0] = pts[1][0] # Top left
			temp_rect[1] = pts[0][0] # Top right
			temp_rect[2] = pts[3][0] # Bottom right
			temp_rect[3] = pts[2][0] # Bottom left

		# If furthest left point is lower than furthest right point,
		# card is tilted to the right
		if pts[1][0][1] > pts[3][0][1]:
			# If card is titled to the right, approxPolyDP returns points
			# in this order: top left, bottom left, bottom right, top right
			temp_rect[0] = pts[0][0] # Top left
			temp_rect[1] = pts[3][0] # Top right
			temp_rect[2] = pts[2][0] # Bottom right
			temp_rect[3] = pts[1][0] # Bottom left
			
		
	maxWidth = wMult * 4
	maxHeight = hMult * 4

	# Create destination array, calculate perspective transform matrix,
	# and warp card image
	dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
	M = cv2.getPerspectiveTransform(temp_rect,dst)
	warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

	

		

	return warp

def process_card(contour, image, convert):
	"""Uses contour to find information about the query card. Isolates rank
	and suit images from the card."""

	# Initialize new Query_card object
	mCard = Magic_card()

	mCard.contour = contour

	# Find perimeter of card and use it to approximate corner points
	peri = cv2.arcLength(contour,True)
	approx = cv2.approxPolyDP(contour,0.01*peri,True)
	pts = np.float32(approx)
	mCard.corner_pts = pts

	# Find width and height of card's bounding rectangle
	x,y,w,h = cv2.boundingRect(contour)
	mCard.width, mCard.height = w, h

	# Find center point of card by taking x and y average of the four corners.
	average = np.sum(pts, axis=0)/len(pts)
	cent_x = int(average[0][0])
	cent_y = int(average[0][1])
	mCard.center = [cent_x, cent_y]

	# Warp card into 200x300 flattened image using perspective transform
	mCard.warp = flattener(image, pts, w, h, convert)

	# TIME TO FIND THE CARD ART



	mCard.art = mCard.warp[40:205, 18:232]
	# y:y+h x:x+w
	# 352 x 252
	mCard.sleeved = mCard.warp[15:336, 11:240]

	return mCard 



def make_record(sig, k=27, N=300):
	"""Makes a record suitable for database insertion.
	"""
	record = []
	signature = sig

	#record.append(signature.tolist())
	record.append(signature)

	words = get_words(signature, k, N)
	max_contrast(words)

	words = words_to_int(words)

	for w in words:
		record.append(w)

	return record


def get_words(array, k, N):
	"""Gets N words of length k from an array.
	"""
	# generate starting positions of each word
	word_positions = np.linspace(0, array.shape[0],
								 N, endpoint=False).astype('int')

	# check that inputs make sense
	if k > array.shape[0]:
		raise ValueError('Word length cannot be longer than array length')
	if word_positions.shape[0] > array.shape[0]:
		raise ValueError('Number of words cannot be more than array length')

	# create empty words array
	words = np.zeros((N, k)).astype('int8')

	for i, pos in enumerate(word_positions):
		if pos + k <= array.shape[0]:
			words[i] = array[pos:pos+k]
		else:
			temp = array[pos:].copy()
			temp.resize(k)
			words[i] = temp

	return words


def words_to_int(word_array):
	"""Converts a simplified word to an integer
	"""
	width = word_array.shape[1]

	# Three states (-1, 0, 1)
	coding_vector = 3**np.arange(width)

	# The 'plus one' here makes all digits positive, so that the
	# integer represntation is strictly non-negative and unique
	return np.dot(word_array + 1, coding_vector)


def max_contrast(array):
	"""Sets all positive values to one and all negative values to -1.
	"""
	array[array > 0] = 1
	array[array < 0] = -1

	return None


def normalized_distance(_target_array, _vec, nan_value=1.0):
	"""Compute normalized distance to many points.
	"""
	target_array = _target_array.astype(int)
	vec = _vec.astype(int)
	topvec = np.linalg.norm(vec - target_array, axis=1)
	norm1 = np.linalg.norm(vec, axis=0)
	norm2 = np.linalg.norm(target_array, axis=1)
	finvec = topvec / (norm1 + norm2)
	finvec[np.isnan(finvec)] = nan_value

	return finvec


def make_dict(l):
	arr = l[0]

	d = dict()

	for li in l[1:]:
		d[li] = arr

	return d

def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape)/2)
	rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
	return result	
