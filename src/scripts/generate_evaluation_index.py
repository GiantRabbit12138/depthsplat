from pathlib import Path
import sys
sys.path.append("..")
from ..evaluation.evaluation_index_generator import (EvaluationIndexGeneratorCfg,
                                                     EvaluationIndexGenerator)

if __name__ == "__main__":
    EvaluationIndexGeneratorCfg.num_target_views = 1
    EvaluationIndexGeneratorCfg.max_distance = 100
    EvaluationIndexGeneratorCfg.min_distance = 0.1
    EvaluationIndexGeneratorCfg.max_overlap = 0.5
    EvaluationIndexGeneratorCfg.min_overlap = 0.1
    EvaluationIndexGeneratorCfg.output_path = Path("../assets")
    EvaluationIndexGeneratorCfg.save_previews = True
    EvaluationIndexGeneratorCfg.seed = 1234
    
    eval_index_generator = EvaluationIndexGenerator(EvaluationIndexGeneratorCfg)
    eval_index_generator.save_index()
