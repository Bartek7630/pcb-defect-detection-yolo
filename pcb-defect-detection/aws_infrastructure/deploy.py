import os
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.session_settings import SessionSettings

os.makedirs("/home/sagemaker-user/sm_tmp", exist_ok=True)

session = sagemaker.Session(
    settings=SessionSettings(local_download_dir="/home/sagemaker-user/sm_tmp")
)
role = sagemaker.get_execution_role()

pytorch_model = PyTorchModel(
    model_data="s3://sagemaker-eu-north-1-YOUR_AWS_ACCOUNT_ID/output/model.tar.gz", 
    role=role,
    entry_point="inference.py",
    source_dir="./pcb_deploy",
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=session,
)

print("Deploying real-time endpoint to AWS SageMaker...")
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="pcb-defect-detector-endpoint-v2",
)
print(f"Endpoint deployed successfully: {predictor.endpoint_name}")