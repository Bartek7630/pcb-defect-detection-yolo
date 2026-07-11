# Automated PCB Defect Detection System (YOLOv8)

## Project Overview
This project implements a computer vision pipeline for automated quality control in electronics manufacturing. Utilizing a fine-tuned **YOLOv8** model, the system detects various anomalies on Printed Circuit Boards in real-time, such as short circuits, missing holes, and mouse bites. 

**Live Interactive Demo:** [[INSERT_HUGGING_FACE_LINK_HERE]](https://huggingface.co/spaces/Bartek7630/pcb_detection)

## Architecture & Deployment Strategy

### Phase 1: AWS Production Infrastructure (Archived)
Initially, the system was designed for enterprise-grade scalability using Amazon Web Services.
* **SageMaker:** The model was deployed as a real-time inference endpoint (`ml.m5.large`).
* **API Gateway & Lambda:** Serverless layers were configured to securely handle external REST HTTP requests, decode Base64 images, and route payloads to the SageMaker endpoint.
* *(Deployment scripts and custom inference code for AWS can be found in the `aws_infrastructure/` directory).*

### Phase 2: FinOps & Serverless Migration (Current)
To optimize costs for continuous portfolio demonstration (avoiding 24/7 SageMaker instance charges), the live architecture was migrated to a zero-cost, serverless environment:
* **Hugging Face Spaces:** Dynamic hardware allocation with Gradio interface.
* **PyTorch Environment Handling:** Implemented dynamic monkey-patching of the `torch.load` mechanism during initialization to bypass restrictive `weights_only` unpickling blockers in strict cloud environments.

## Tech Stack
* **Deep Learning:** PyTorch, Ultralytics YOLOv8
* **Cloud (AWS):** SageMaker, S3, API Gateway, AWS Lambda
* **Deployment:** Hugging Face Spaces, Gradio
* **Languages:** Python 3.10+

## How to Run Locally
1. Clone the repository: `git clone https://github.com/Bartek7630/pcb-defect-detection.git`
2. Install dependencies: `pip install -r huggingface_app/requirements.txt`
3. Run the Gradio interface: `python huggingface_app/app.py`
