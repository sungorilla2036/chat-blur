import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en', use_gpu=False)
frame_number = 0

input_filename = 'path/to/input_video.mp4'
output_filename = 'path/to/output_video.mp4'

input_video = cv2.VideoCapture(input_filename)
start_time_seconds = 0
fps = input_video.get(cv2.CAP_PROP_FPS)
input_video.set(cv2.CAP_PROP_POS_FRAMES, int(fps * start_time_seconds))
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourccCode = 'MP4V'

output_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*fourccCode), fps, (width, height))

# print properties
print('width: ', width)
print('height: ', height)
print('fps: ', fps)
print('fourcc: ', fourccCode)

while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break

    # text area to run OCR on. Defaults are: (0,0) is top left corner. (width, height) is bottom right corner
    textarea_y_start = 0
    textarea_y_end = height
    textarea_x_start = 0
    textarea_x_end = width

    textarea = frame[textarea_y_start:textarea_y_end, textarea_x_start:textarea_x_end]
    pred_data = ocr.ocr(textarea, rec=False, cls=False)
    print(pred_data)
    for bounding_box in pred_data[0]:
        #blur area inside bounding box
        blur = cv2.GaussianBlur(textarea[int(bounding_box[0][1]):int(bounding_box[2][1]), int(bounding_box[0][0]):int(bounding_box[2][0])], (51,51), 0)
        
        #paste blurred area back
        textarea[int(bounding_box[0][1]):int(bounding_box[2][1]), int(bounding_box[0][0]):int(bounding_box[2][0])] = blur

        #textarea = cv2.rectangle(textarea, (int(bounding_box[0][0]), int(bounding_box[0][1])), (int(bounding_box[2][0]),int(bounding_box[2][1])), (36,255,12), 1)
    frame[textarea_y_start:textarea_y_end, textarea_x_start:textarea_x_end] = textarea

    # cv2.imshow('frame', frame)
    # Press Q on keyboard to  exit
    # if cv2.waitKey() & 0xFF == ord('q'):
    #     break

    frame_number += 1
    # print progress

    output_video.write(frame)

input_video.release()
output_video.release()
cv2.destroyAllWindows()