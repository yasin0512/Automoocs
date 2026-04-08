"""MOOCs Auto-Generation System — Backend Modules"""
from .preprocessing    import AudioProcessor, VideoSlicer
from .action_modeling  import (PoseAnalyzer, ActionLabeler, SpeechRateCalc,
                                MultimodalActionModel, ActionModelTrainer,
                                ActionInference, run_full_training_pipeline)
from .text_cleaner     import TextCleaner
from .syllabus_aligner import SyllabusAligner
from .ppt_generator    import PPTGenerator, StableDiffusionGenerator, segment_slides
from .pipeline         import ScriptGenerator, VoiceVideoGenerator, MOOCsPipeline

__all__ = [
    "AudioProcessor","VideoSlicer",
    "PoseAnalyzer","ActionLabeler","SpeechRateCalc",
    "MultimodalActionModel","ActionModelTrainer","ActionInference",
    "run_full_training_pipeline",
    "TextCleaner","SyllabusAligner",
    "PPTGenerator","StableDiffusionGenerator","segment_slides",
    "ScriptGenerator","VoiceVideoGenerator","MOOCsPipeline",
]
