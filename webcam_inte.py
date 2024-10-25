from ultralytics import YOLO
import cv2
import pyttsx3


engine = pyttsx3.init()

stairs_model = YOLO('training_results/best.pt')    
potholes_model = YOLO('training_results/bestp.pt')  


cap = cv2.VideoCapture(0)

previous_steps_detected = False
previous_pothole_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    stairs_results = stairs_model.predict(source=frame, conf=0.8, show=False, save=False, save_txt=False)
    stairs_detected = len(stairs_results[0]) > 0 

    potholes_results = potholes_model.predict(source=frame, conf=0.9, show=False, save=False, save_txt=False)
    pothole_detected = len(potholes_results[0]) > 0  

    stairs_annotated_frame = stairs_results[0].plot() 
    potholes_annotated_frame = potholes_results[0].plot()  

    combined_frame = cv2.addWeighted(stairs_annotated_frame, 0.5, potholes_annotated_frame, 0.5, 0)

    if stairs_detected and not previous_steps_detected:
        engine.say("Steps detected.")
        engine.runAndWait()
        previous_steps_detected = True 

    elif not stairs_detected:
        previous_steps_detected = False  

   
    if pothole_detected and not previous_pothole_detected:
        engine.say("Pothole detected.")
        engine.runAndWait()
        previous_pothole_detected = True  
    elif not pothole_detected:
        previous_pothole_detected = False 

    
    cv2.imshow('Stairs and Potholes Detection', combined_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 
cap.release()
cv2.destroyAllWindows()
