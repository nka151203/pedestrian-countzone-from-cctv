# CountZone: Applications for counting the number of people appearing in your custom ROI (Region of Interest)

## 1. Overview

**Scenario:** CountZone - The software is developed to assist users in monitoring and counting the number of people appearing within a specific Region of Interest (ROI) in a video. Tracking the number of people in an ROI plays a significant role in analyzing how attractive a particular area in the CCTV. For example, the number of people interested in a booth at a supermarket can reveal local consumer preferences; or monitoring which entertainment zones attract the most attention in a shopping mall can offer insights into public interest. These applications inspired me to create a small-scale software solution that allows users to count the number of people within a custom-defined ROI.

---
## 2. Dataset and Tech
**Dataset**: MOT-20 Challenge https://motchallenge.net/data/MOT20/\
**YOLOv11**: A famous framework for detection https://docs.ultralytics.com/vi/models/yolo11/ \
**Boostrack++**: A SOTA tracking Algorithm https://github.com/vukasin-stanojevic/BoostTrack 

---
## 3. Project Structure
*My software is built based on two main stages:*

**1. YOLOv11 Fine-tuning**: The first stage involves fine-tuning the YOLOv11 model on the MOT20 dataset — a well-known dataset widely used in the MOT Challenges. This dataset is highly suitable for our problem because it consists of fixed-camera video segments (similar to CCTV footage) with dense pedestrian scenes. As a result, the fine-tuned YOLOv11 model becomes highly effective at detecting crowded human regions.

**2. Tracking with BoostTrack++**: Once the fine-tuned YOLOv11 model is ready, the second stage involves applying BoostTrack++, a state-of-the-art tracking algorithm as of 2025. BoostTrack++ is used to track people throughout the video, leveraging the detections provided by our customized YOLOv11 model.  

```bash
pedestrian-countzone-from-cctv/
│
├── detecting/          # Preparing for training
├── model/              # Save fine-tuned model 
├── tracking/           # Tracking with Boosttrack++
```
---
## 4. How to get started

```bash
git clone https://github.com/nka151203/pedestrian-countzone-from-cctv.git
pip install -r requirements.txt
cd pedestrian-countzone-from-cctv
```

### 4.1 Try fine-tuning YOLOv11 Model (Optional)
**Note**: You can ignore this step because I did help you. Let's check `model/weights/best.pt`
```bash
python detecting/train.py
```
### 4.2 Run our application 
```bash
cd tracking/boosttrack
python gui.py
```

## 📌 Notes

- Make sure you installed Python 3.10 or higher
- Your computer is alive :)
- The speed of tracking on the application is proportional to your computer configuration.

---
## 🌚 Limitations and future work

- The current user interface remains quite simple, focusing solely on counting the number of people within a defined ROI, without further feature development for broader application use.

- In some camera setups that differ significantly from the MOT dataset (e.g., variations in camera angles, contrast levels, or cameras being too close or too far), the model may not perform as accurately as expected.
- Tracking Algorithm still aren't set up on CUDA, so tracking is quite slow because it's running on CPU

- In the future, I hope to improve the generalization capabilities of the model to handle a wider range of camera scenarios more effectively. Additionally, I aim to expand beyond basic counting by exploring more advanced tasks such as anomaly detection, multi-camera object tracking, and measuring engagement levels — for example, by calculating the duration a person stays within a specific area for analystic.

---

## 📬 Contact
**If there is any copyright issue or you need help, please contact:**
- Author: Nguyen Khac An
- Email: nka151203@gmail.com
