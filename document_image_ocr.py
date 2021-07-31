# import the necessary packages
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2


def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll align to template")
args = vars(ap.parse_args())

# create a named tuple which we can use to create locations of the
# input document which we wish to OCR
OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
	"filter_keywords"])
# define the locations of each area of the document we wish to OCR
OCR_LOCATIONS = [
	OCRLocation("title", (124, 79, 100, 21),
		["1", "."]),
	OCRLocation("sub_title", (149, 131, 132, 18),
		["1", "."]),
	OCRLocation("sub_text", (104, 181, 504, 95),
		["[","]","delete"])
]

# load the input image and template from disk
print("[INFO] loading images...")
# image = io.imread(args["image"])
image = cv2.imread(args["image"])

# initialize a results list to store the document OCR parsing results
print("[INFO] OCR'ing document...")

parsingResults = []
# loop over the locations of the document we are going to OCR
for loc in OCR_LOCATIONS:
	# extract the OCR ROI from the aligned image
	(x, y, w, h) = loc.bbox
	roi = image[y:y + h, x:x + w]

	# cv2.imshow("Input", imutils.resize(roi, width=700))
	# cv2.waitKey(0)

	# OCR the ROI using Tesseract
	rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	text = pytesseract.image_to_string(rgb)

	# break the text into lines and loop over them
	for line in text.split("\n"):
		# if the line is empty, ignore it
		if len(line) == 0:
			continue
		# convert the line to lowercase and then check to see if the
		# line contains any of the filter keywords (these keywords
		# are part of the *form itself* and should be ignored)
		lower = line.lower()
		count = sum([lower.count(x) for x in loc.filter_keywords])
		# if the count is zero then we know we are *not* examining a
		# text field that is part of the document itself (ex., info,
		# on the field, an example, help text, etc.)
		if count == 0:
			# update our parsing results dictionary with the OCR'd
			# text if the line is *not* empty
			parsingResults.append((loc, line))

# initialize a dictionary to store our final OCR results
results = {}
# loop over the results of parsing the document
for (loc, line) in parsingResults:
	# grab any existing OCR result for the current ID of the document
	r = results.get(loc.id, None)
	# if the result is None, initialize it using the text and location
	# namedtuple (converting it to a dictionary as namedtuples are not
	# hashable)
	if r is None:
		results[loc.id] = (line, loc._asdict())
	# otherwise, there exists an OCR result for the current area of the
	# document, so we should append our existing line
	else:
		# unpack the existing OCR result and append the line to the
		# existing text
		(existingText, loc) = r
		text = "{}\n{}".format(existingText, line)
		# update our results dictionary
		results[loc["id"]] = (text, loc)

# loop over the results
for (locID, result) in results.items():
	# unpack the result tuple
	(text, loc) = result
	# display the OCR result to our terminal
	print(loc["id"])
	print("=" * len(loc["id"]))
	print("{}\n\n".format(text))
	# extract the bounding box coordinates of the OCR location and
	# then strip out non-ASCII text so we can draw the text on the
	# output image using OpenCV
	(x, y, w, h) = loc["bbox"]
	clean = cleanup_text(text)
	# draw a bounding box around the text
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# loop over all lines in the text
	# for (i, line) in enumerate(text.split("\n")):
	# 	# draw the line on the output image
	# 	startY = y + (i * 70) + 40
	# 	cv2.putText(image, line, (x, startY),
	# 		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 0)

# show the input and output images, resizing it such that they fit
# on our screen
cv2.imshow("Input", imutils.resize(image, width=700))
cv2.waitKey(0)