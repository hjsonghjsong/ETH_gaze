# hjsong
# I have two prints (pitch roll yaw) 
# first one for normalized image, and the second one for denormalized one 
# Wiam, pleaes print both sets for the video

# the location where is the input file : refer to f1 var


import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network
import math

#for profiling 
import cProfile
import re

from head_pose import HeadPoseEstimator
from utils import pitchyaw_to_vector, vector_to_pitchyaw

#Constant parts hjsong 
#f1 = '../Data/bilateralLoss.MOV' # patient video
f1='../Data/AG, portrait, 28_, with side light.MOV'  #which video you look into 
buffer_size=40   # could increase buffer_size as you wish

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# from https://learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R)) #hjsong for temp
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])



def calculateHeadPoseinCC ( rotation_vector, translation_vector, camera_matrix, dist_coeffs) :
    axis = np.float32([[500,0,0], [0,500,0],  [0,0,500]])
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    
    '''
    cv2.line(img, noseTipin2D, tuple(imgpts[1].astype(int).ravel()), (0,255,0), 3) #GREEN
    cv2.line(img, noseTipin2D, tuple(imgpts[0].astype(int).ravel()), (255,0,), 3) #BLUE
    cv2.line(img, noseTipin2D, tuple(imgpts[2].astype(int).ravel()), (0,0,255), 3) #RED
    '''
    return pitch, roll, yaw

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def DeNormalizePitchYaw(fIndex,pitchyaw, R) : 
    #normalized one
    #print pitch roll yaw   pitchyaw[0] : pitch, pitchyaw [1] : yaw hjsong 
    print ( f'{fIndex}' + "  "+   f'{np.degrees(pitchyaw[0]):5.2f}' + "  "  +  "0" + "  "  + f'{np.degrees(pitchyaw[1]):5.2f}') 
        
    
    #convert to 3D gaze vector
    GazeVectorin3D=pitchyaw_to_vector(pitchyaw)
    GazeVectorin3D=np.reshape(GazeVectorin3D, (3,-1))
    #Calculate R.I
    try:
        inverse = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        pass
    else:
        DeNormGazeVectorin3D=np.dot (inverse, GazeVectorin3D)  #Denormalize
        GazeVectorin3D=np.reshape(GazeVectorin3D, (-1,3))  #hjsong do i have to scale back and translate to face center?
        outPitchYaw=vector_to_pitchyaw (DeNormGazeVectorin3D)
        print ( f'{fIndex}' + "  "+   f'{np.degrees(outPitchYaw[0][0]):5.2f}' + "  "  +  "0" + "  "  + f'{np.degrees(outPitchYaw[0][1]):5.2f}')    


def draw_gaze( fIndex, image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    #hjsong

    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])

    #print ( f'{fIndex}' + "  "+   f'{dx}' + "  "  +    f'{dy}')    


    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model  #hjsong face_model in 3D coordinate  shape : (3,6)
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]  # shpe (3,)
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped, R

if __name__ == '__main__':

    #hjsong
    cap = cv2.VideoCapture(f1)
    #cap = cv2.VideoCapture(0)  # from webcam
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    bi=-1
    ret, image = cap.read()
    size = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX 

    imgBuffer = np.arange (buffer_size* image.shape[0]*image.shape[1]*image.shape[2],  dtype=np.uint8).reshape(-1, image.shape[0], image.shape[1], image.shape[2])

    #hjsong
    fIndex=-1   # frame index
    startFrame = 0
    (h, w) = image.shape[:2]
    
    
    while True:
        bi = (bi + 1 ) %  buffer_size  # buffer index 
        imgBuffer[bi] = image

        ret, image = cap.read()
        if (ret==False) :
            break  
       
        
        fIndex+=1      
        if (fIndex<startFrame) :
                print ("findex"+str(fIndex))
                continue
        
        #cv2.putText(image, 'fIndex'+str(fIndex), (90, 30), font, 2, (255, 255, 128), 3) 



        ''' hjsong 
        img_file_name = './example/input/cam00.JPG'
        print('load input face image: ', img_file_name)
        image = cv2.imread(img_file_name)
        '''

        predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')  #hjsong https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
        # face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')
        face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
        detected_faces = face_detector(image, 1)
        if len(detected_faces) == 0:
            print('warning: no detected face')
            exit(0)
        #print('detected one face')
        shape = predictor(image, detected_faces[0]) ## only use the first detected face (assume that each input image only contains one face)
        shape = face_utils.shape_to_np(shape)
        landmarks = []
        for (x, y) in shape:
            landmarks.append((x, y))
        landmarks = np.asarray(landmarks)

        #draw landmarks 


        # load camera information
        cam_file_name = './example/input/cam00.xml'  # this is camera calibration information file obtained with OpenCV
        if not os.path.isfile(cam_file_name):
            print('no camera calibration file is found.')
            exit(0)
        fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
        camera_distortion = fs.getNode('Distortion_Coefficients').mat()

        #print('estimate head pose')
        # load face model
        face_model_load = np.loadtxt('face_model.txt')  # Generic face model with 3D facial landmarks
        landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
        face_model = face_model_load[landmark_use, :]
        # estimate the head pose,
        ## the complex way to get head pose information, eos library is required,  probably more accurrated
        # landmarks = landmarks.reshape(-1, 2)
        # head_pose_estimator = HeadPoseEstimator()
        # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
        ## the easy way to get head pose information, fast and simple
        facePts = face_model.reshape(6, 1, 3)
        landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
        landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
        landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
        hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

        ###################################################################
        #Method 0 to convert to euler angles
        #https://github.com/neoronbug/Roll-Yaw-Pitch-Angles/blob/master/pose_estimation.py, Wiam
        #pitch, roll, yaw=calculateHeadPoseinCC (hr, ht, camera_matrix, camera_distortion) 
        #imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        ###################################################################

        #draw the line 
        axis = np.float32([[500,0,0], [0,500,0],  [0,0,500]])
        srcin3D = np.float32([0,0,0])
        imgpts, jac = cv2.projectPoints(axis, hr, ht, camera_matrix, camera_distortion)
        srcin2D, jac = cv2.projectPoints(srcin3D, hr, ht, camera_matrix, camera_distortion)
        srcin2D=(int(tuple(srcin2D[0][0])[0]), int(tuple(srcin2D[0][0])[1]))
        #srcin2D=(1023, 542)
        cv2.line(image, srcin2D , tuple(imgpts[1].astype(int).ravel()), (0,255,0), 3) #GREEN  #landmarks[30] nosetips
        cv2.line(image, srcin2D, tuple(imgpts[0].astype(int).ravel()), (255,0,), 3) #BLUE
        cv2.line(image, srcin2D, tuple(imgpts[2].astype(int).ravel()), (0,0,255), 3) #RED

        
        ###################################################################
        #Method 1 to convert to euler angles you can use either the above or the below : from learnopencv.com
        rvec_matrix = cv2.Rodrigues(hr)[0]  
        pitch, roll, yaw= rotationMatrixToEulerAngles(rvec_matrix)
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        t=roll
        roll=-int(yaw)
        yaw=-t
        ###################################################################
        

        
        #print ( f'{fIndex}' + "  "+   f'{pitch:5.2f}' + "  "  +  f'{roll:5.2f}'+ "  "  + f'{yaw:5.2f}')    
        #cv2.putText(image, 'frame :' + str(fIndex) + " yaw:" +  str(int(yaw)),  (50, 50), font, 2, (255, 255, 128), 3)
        #cv2.moveWindow('img', 100,30)
        #cv2.imshow('img', image)

      
        # data normalization method
        # print('data normalization, i.e. crop the face image')
        # R: rotation matrix used for normalization 
        img_normalized, landmarks_normalized, R = normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)

        
        #GAZE hjsong
        #print('load gaze estimator')
        model = gaze_network()
        #model.cuda() # comment this line out if you are not using GPU
        pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
        if not os.path.isfile(pre_trained_model_path):
            print('the pre-trained gaze estimation model does not exist.')
            exit(0)
        else:
            #print('load the pre-trained model: ', pre_trained_model_path)
            pass
        ckpt = torch.load(pre_trained_model_path,  map_location=torch.device('cpu'))  #hjsong
        model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
        model.eval()  # change it to the evaluation mode
        input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
        input_var = trans(input_var)
        input_var = torch.autograd.Variable(input_var.float()) #hjsong .cuda())
        input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
        pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
        pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
        pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
        
        #print('prepare the output')
        # draw the facial landmarks
        landmarks_normalized = landmarks_normalized.astype(int) # landmarks after data normalization
        for (x, y) in landmarks_normalized:
            cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img_normalized, 'fIndex'+str(fIndex), (30, 50), font, 1, (255, 255, 128), 3) #hjsong
        cv2.imshow('img normalized', img_normalized)
        
        #hjsong #
        DeNormalizePitchYaw(fIndex,pred_gaze_np, R)
        
        face_patch_gaze = draw_gaze(fIndex, img_normalized, pred_gaze_np )  # draw gaze direction on the normalized face image
        #output_path = 'example/output/results_gaze.jpg'
        #print('printing output')
        #cv2.imwrite(output_path, face_patch_gaze)
       
        cv2.waitKey(1)

    cv2.destroyAllWindows()