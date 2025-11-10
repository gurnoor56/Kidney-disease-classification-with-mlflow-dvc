from KidneyClassification import logger
from KidneyClassification.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from KidneyClassification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from KidneyClassification.pipeline.stage_03_model_training import ModelTrainingPipeline   
from KidneyClassification.pipeline.stage_04_model_evaluation import EvaluationPipeline  



STAGE_NAME="Data Ingestion Stage"
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    

STAGE_NAME="Prepare Base Model Stage"
if __name__ == "__main__":
    try:
        logger.info(f"****************")    
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME="Training"
if __name__ == "__main__":
    try:
        logger.info(f"************")
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Evaluation stage"
if __name__ == "__main__":
    try:
        logger.info(f"***************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj=EvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=========x")
    except Exception as e:
        logger.exception(e)
        raise e