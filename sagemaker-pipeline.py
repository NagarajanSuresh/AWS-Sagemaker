import time
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep,TrainingStep
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from sagemaker.workflow.steps import TuningStep

from sagemaker.estimator import Estimator

pipeline_session = PipelineSession()

ROLE_FOR_PROCESSING_JOB = "arn:aws:iam::853973692277:role/ns2312-lr-pipeline-execution-role"
PROCESSING_JOB_CUSTOM_IMAGE_URI = "853973692277.dkr.ecr.us-east-1.amazonaws.com/ns2312-lr-preprocessing:latest"

processor = Processor(
    image_uri=PROCESSING_JOB_CUSTOM_IMAGE_URI,
    instance_count=1,
    role=ROLE_FOR_PROCESSING_JOB,
    instance_type="ml.m5.large",
    sagemaker_session=pipeline_session
)

step_preprocess = ProcessingStep(
    name="HousingPreprocessing",
    processor=processor,
    inputs=[
        ProcessingInput(
            source="s3://ns2312-lr-model-bucket/Housing.csv",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/train",
            destination="s3://ns2312-lr-model-bucket/HousingDataset/Training"
        ),
        ProcessingOutput(
            output_name="val_data",
            source="/opt/ml/processing/val",
            destination="s3://ns2312-lr-model-bucket/HousingDataset/Validation"
        ),
        ProcessingOutput(
            output_name="test_data",
            source="/opt/ml/processing/test",
            destination="s3://ns2312-lr-model-bucket/HousingDataset/Test"
        ),
        ProcessingOutput(
            output_name="scaler",
            source="/opt/ml/processing/scaler",
            destination="s3://ns2312-lr-model-bucket/scaler"
        )
    ],
    job_arguments=[
        "--input-dir", "/opt/ml/processing/input",
        "--csv-name", "Housing.csv",
        "--train-dir", "/opt/ml/processing/train",
        "--val-dir", "/opt/ml/processing/val",
        "--test-dir", "/opt/ml/processing/test"
    ]
)

ROLE_FOR_TRAINING_JOB = "arn:aws:iam::853973692277:role/ns2312-lr-pipeline-execution-role"
TRAINING_JOB_CUSTOM_IMAGE_URI = "853973692277.dkr.ecr.us-east-1.amazonaws.com/ns2312-lr-training:latest"
MODEL_ARTIFACT_OUTPUT_S3_URI = "s3://ns2312-lr-model-bucket/model_output/"

estimator = Estimator(
    image_uri=TRAINING_JOB_CUSTOM_IMAGE_URI,
    role=ROLE_FOR_TRAINING_JOB,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=MODEL_ARTIFACT_OUTPUT_S3_URI,
    sagemaker_session=pipeline_session
)
estimator.set_hyperparameters(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=150
)

step_train = TrainingStep(
    name="HousingTraining",
    estimator=estimator,
    inputs={
        "train": step_preprocess.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
        "val": step_preprocess.properties.ProcessingOutputConfig.Outputs["val_data"].S3Output.S3Uri
    }
)

TRAINING_JOB_CUSTOM_IMAGE_URI = "853973692277.dkr.ecr.us-east-1.amazonaws.com/ns2312-lr-training:latest"
TRAINING_DS_S3_URI = "s3://ns2312-lr-model-bucket/HousingDataset/Training/"
VALIDATION_DS_S3_URI = "s3://ns2312-lr-model-bucket/HousingDataset/Validation/"

ROLE_FOR_HP_TUNING_JOB = "arn:aws:iam::853973692277:role/ns2312-lr-pipeline-execution-role"
timestamp = int(time.time())

estimator = Estimator(
    image_uri=TRAINING_JOB_CUSTOM_IMAGE_URI,
    role=ROLE_FOR_HP_TUNING_JOB,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=MODEL_ARTIFACT_OUTPUT_S3_URI,
    base_job_name=f"ns2312-housing-training-job-{timestamp}"
)
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.01, 0.3),
    "max_depth": IntegerParameter(3, 10),
    "n_estimators": IntegerParameter(50, 300),
}

objective_metric_name = "validation:rmse"

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[
        {"Name": "validation:rmse", "Regex": "validation:rmse=([0-9\\.]+)"}
    ],
    max_jobs=6,
    max_parallel_jobs=2,
    objective_type="Minimize",
    base_tuning_job_name=f"ns2312-hpo-{timestamp}"
)

tuning_step = TuningStep(
    name="HyperparameterTuning",
    tuner=tuner,
    inputs={
        "train": step_preprocess.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
        "val": step_preprocess.properties.ProcessingOutputConfig.Outputs["val_data"].S3Output.S3Uri,
    }
)


pipeline = Pipeline(
    name="Ns2312HousingPipeline",
    steps=[step_preprocess,tuning_step],
    sagemaker_session=pipeline_session
)

pipeline.upsert(role_arn=ROLE_FOR_PROCESSING_JOB)
execution = pipeline.start()
print("Pipeline execution started:", execution.arn)