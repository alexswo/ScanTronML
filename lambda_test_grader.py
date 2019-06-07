import boto3
import numpy as np
import argparse
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import cv2
import urllib.parse
import json
import uuid
import base64
from decimal import Decimal

def lambda_handler(event, context):
    
    print("[INFO] boto version: {}".format(boto3.__version__))
    s3 = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
    exam = dynamodb.Table('Exam')
    grade = dynamodb.Table('Grade')
    client = boto3.client('lambda')
    
    
    # Get S3 bucket and image file name
    bucket = event['Records'][0]['s3']['bucket']['name']
    image_file_name = urllib.parse.unquote(event['Records'][0]['s3']['object']['key'])
    
    # Parse the image's file name
    parsed_file_name = image_file_name.split('/')
    user_id = parsed_file_name[1].split('=')[1]
    exam_id = parsed_file_name[2].split('=')[1]
    image_id = parsed_file_name[3]
    
    print("[INFO] Image File Name {}".format(image_file_name))
    
    # Get json response for where the object is situated
    file_obj = s3.get_object(Bucket = bucket, Key = image_file_name)
    
    # Read in binary object
    file_content = file_obj['Body'].read()
    
    np_array = np.fromstring(file_content, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
     
    # define the answer key which maps the question number
    # to the correct answer
    try:
        response = exam.get_item(
            Key={
                'userid': user_id,
                'examid': exam_id
            }
        )
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        answer_key = [int(ord(v.lower())-ord('a')) for v in response['Item']['answers']]
        print("[INFO] Answer key {}".format(answer_key))
        
    
    # load the image, convert it to grayscale, blur it
    # slightly, then find edges
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None
     
    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
     
        # loop over the sorted contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
     
            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break
    
    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    # cv2.imshow("paper", paper)
    # cv2.imwrite("img/paper.png", paper)
    # apply Otsu's thresholding method to binarize the warped
    # piece of paper
    thresh = cv2.threshold(warped, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # h, w  = thresh.shape
    # crop_img = paper[:int(h/8), int(w/9):]
    # crop_img = cv2.imencode('.jpg', file_content)[1].tostring()
    payload = {
        'bucket': bucket, 
        'image_file_name': image_file_name
    }
    response = client.invoke(
        FunctionName='get_student_id',
        InvocationType='RequestResponse',
        Payload=bytes(json.dumps(payload), 'utf-8')
    )
    
    response_load = json.loads(response['Payload'].read().decode('utf8'))
    print('[INFO] payload: {}'.format(response_load))
    
    student_id = response_load['student_id']
    print('[INFO] student_id: {}'.format(student_id))
    
    # s3.put_object(Bucket=bucket, Key='asdasdasdads', Body=crop_img)
        
    # cv2.imshow("thresh", thresh)
    # cv2.imwrite("img/thresh.png", thresh)
    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
     
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
     
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)
    
    # sort the question contours top-to-bottom, then initialize
    # the total number of correct answers
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0
    
    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None
    
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
     
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
     
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
    
        # initialize the contour color and the index of the
        # *correct* answer
        color = (0, 0, 255)
        k = answer_key[q]
    
        # check to see if the bubbled answer is correct
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
    
        # draw the outline of the correct answer on the test
        cv2.drawContours(paper, [cnts[k]], -1, color, 3)
    
    # grab the test taker
    score = (correct / len(answer_key)) * 100
    print("[INFO] score: {:.2f}%".format(score))
    cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    # convert from numpy array to binary string
    paper = cv2.imencode('.jpg', paper)[1].tostring()
    # cv2.imshow("Original", image)
    # cv2.imshow("Exam", paper)
    
    # store graded answer sheets back to s3 bucket
    new_file_name = 'graded/'+'/'.join(parsed_file_name[1:])
    
    s3.put_object(Bucket=bucket, Key=new_file_name, Body=paper)
    bucket_url = 'scantron-answer-sheets.s3-us-west-2.amazonaws.com'
    
    grade_id = str(uuid.uuid4())
    # add grade object into DynamoDB
    grade_params = {
        'userid': user_id,
        'studentid': student_id, 
        'gradeid': grade_id,
        'examid': exam_id,
        'score': Decimal(score),
        'graded_url': '{}/{}'.format(bucket_url, new_file_name),
        'raw_url': '{}/{}'.format(bucket_url, image_file_name)
    }
               
               
    exam.update_item(Key={
        'userid': user_id,
        'examid': exam_id
    }, 
    UpdateExpression='ADD gradeids :x',
    ExpressionAttributeValues={
        ':x': set([grade_id])
    })
    grade.put_item(Item=grade_params)