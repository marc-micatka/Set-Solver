from card_class import *
import cv2


def run_live_vid():
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.75
    fontColor = (0, 0, 255)
    lineType = 2
    rectLine = 3

    # Start video feed from webcam
    cap = cv2.VideoCapture(2)

    # Set frame width and height to HD resolution (1280x720)
    w, h = 1280, 720
    cap.set(3, 1280)
    cap.set(4, 720)

    cv2.namedWindow('result')

    x_rect = 10
    y_rect = 50
    w_rect = w - x_rect
    h_rect = h - 10

    pt1 = (x_rect, y_rect)
    pt2 = (w - x_rect, h - 10)

    y_min, y_max = y_rect, h - 10
    x_min, x_max = x_rect, w - x_rect

    _, frame = cap.read()
    img = frame[y_min+rectLine:y_max-rectLine, x_min+rectLine:x_max-rectLine]

    # Count Cards in SMALL SCENE
    image_small = img[::2, ::2]
    scene = ProcessScene(image_small)
    scene.process(only_count=True)
    card_count = scene.card_count

    avg1 = np.float32(img)


    frame_count = 0
    start_time = time.time()

    n_frames = []
    scenes = []
    old_frame = img

    thresh_buffer = []
    buffer_size = 5

    motion = False
    scene_processed = False
    scene_res = img

    while 1:
        _, frame = cap.read()
        img = frame[y_min+rectLine:y_max-rectLine, x_min+rectLine:x_max-rectLine]

        cv2.accumulateWeighted(img, avg1, .5)
        res1 = cv2.convertScaleAbs(avg1)
        diff = cv2.subtract(res1, old_frame)

        image_small = img[::2, ::2, :]
        diff_small = diff[::2, ::2, :]

        diff_bw = bw_filter(diff_small)
        thresh_diff = minmaxNorm(threshold_img(diff_bw))
        thresh_diff = thresh_diff.mean()

        # Empty Buffer - process the scene
        if len(thresh_buffer) == 0:
            scene = ProcessScene(img)
            scene.process()
            scene_res = scene.draw_symbols(draw_fill=True, draw_num=True, draw_color=True, draw_shape=True)
            game = PlayGame(res1, [scene])
            game.solve()
            scene_res = game.draw_solution()
            card_count = scene.card_count

            scene_processed = True

        if len(thresh_buffer) < buffer_size:
            thresh_buffer.append(thresh_diff)
        else:
            thresh_buffer.append(thresh_diff)
            thresh_buffer.pop(0)

        if np.all(np.array(thresh_buffer) < 5e-4):
            motion = False
        else:
            motion = True

        # PLAY GAME
        if not motion and not scene_processed:
            scene = ProcessScene(img)
            scene.process()
            scene_res = scene.draw_symbols(draw_fill=True, draw_num=True, draw_color=True, draw_shape=True)
            game = PlayGame(res1, [scene])
            game.solve()
            scene_res = game.draw_solution()

            txt = "Card Count: " + str(scene.card_count)
            scene_processed = True

        if not motion and scene_processed:
            txt = "Card Count: " + str(scene.card_count)

        else:
            txt = "Motion Detected..."
            scene_processed = False

        new_frame = np.zeros_like(frame)
        cv2.putText(new_frame, txt, (10, 25), font, fontScale, fontColor, lineType)

        # Graphics
        #######################################
        # Display Game Field Rectange
        cv2.rectangle(new_frame, pt1, pt2, GREEN, thickness=rectLine)

        # Display FPS
        frame_count = frame_count + 1
        end_time = time.time() - start_time
        fps = 'FPS: {0:.1f}'.format(1 / (end_time / frame_count))

        pos = (x_max - 200, 25)
        cv2.putText(frame, fps, pos, font, fontScale, fontColor, lineType)
        ###################################################

        # print(scene_res.shape, frame.shape, y_min+rectLine,y_max-rectLine,x_min+rectLine,x_max-rectLine)

        new_frame[y_min+rectLine:y_max-rectLine, x_min+rectLine:x_max-rectLine] = scene_res
        output = new_frame
        old_frame = img

        '''scene = ProcessScene(res1)
        scene.process()
    
        # output= scene.draw_symbols(draw_fill=True, draw_num=True, draw_color=True, draw_shape=True)
        n_frames.append(res1)
        scenes.append(scene)
        output = res1
    
        # if not solution_found:
        if frame_count >= n:
    
            # After n frames, start looking for solution
            n_frames.pop(0)
            scenes.pop(0)
            game = PlayGame(res1, scenes)
            game.solve()
    
            # if game.solved:
    
            output = game.draw_solution()
        frame_count += 1'''

        cv2.imshow('result', output)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # img_array = setup_imgs(TRAIN_DIR)
    # process_image(img_array[0])
    run_live_vid()
