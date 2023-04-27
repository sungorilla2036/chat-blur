import argparse
import cv2
from paddleocr import PaddleOCR

def blur_text_in_video(video_path, x1, y1, x2, y2, use_gpu=False, output_path='output.mp4'):
    ocr = PaddleOCR(lang='en', use_gpu=use_gpu)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourccCode = 'MP4V'
    fourcc = cv2.VideoWriter_fourcc(*fourccCode)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if x1 is None or y1 is None or x2 is None or y2 is None:
        x1 = 0
        y1 = 0
        x2 = width
        y2 = height

    # print properties
    print('width: ', width)
    print('height: ', height)
    print('fps: ', fps)
    print('fourcc: ', fourccCode)
    while cap.isOpened():
        ret, frame = cap.read()
        textarea = frame[y1:y2,x1:x2]
        if ret:
            pred_data = ocr.ocr(textarea, rec=False, cls=False)
            print(pred_data)
            for bounding_box in pred_data[0]:
                #blur area inside bounding box
                blur = cv2.GaussianBlur(textarea[int(bounding_box[0][1]):int(bounding_box[2][1]), int(bounding_box[0][0]):int(bounding_box[2][0])], (51,51), 0)
                
                #paste blurred area back
                textarea[int(bounding_box[0][1]):int(bounding_box[2][1]), int(bounding_box[0][0]):int(bounding_box[2][0])] = blur

                #textarea = cv2.rectangle(textarea, (int(bounding_box[0][0]), int(bounding_box[0][1])), (int(bounding_box[2][0]),int(bounding_box[2][1])), (36,255,12), 1)
            frame[y1:y2,x1:x2] = textarea
            out.write(frame)

            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blur text in a video')
    parser.add_argument('video_path', type=str,
                        help='path to input video file')
    parser.add_argument('--bounding_rect', type=str, default=None,
                        help='coordinates of bounding box as a single string')
    parser.add_argument('--use_gpu', action='store_true',
                        help='use GPU for OCR (default: False)')
    parser.add_argument('--output_path', type=str,
                        help='path to output video file (default: output.mp4)',
                        default='output.mp4')
    args = parser.parse_args()

    if args.points is not None:
        points = args.points.split()
        x1, y1, x2, y2 = map(int, points)
    else:
        x1 = y1 = x2 = y2 = None

    blur_text_in_video(args.video_path, x1, y1, x2, y2,
                       use_gpu=args.use_gpu, output_path=args.output_path)