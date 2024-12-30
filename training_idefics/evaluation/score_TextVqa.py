# Copyright (c) Facebook, Inc. and its affiliates.
import re
import json
from tqdm import tqdm
from utils import EvalAIAnswerProcessor


# TextVQA scoring

class TextVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        """
        compute the accuracy (soft score) of human answers
        """
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        unique_answer_scores = {}

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [
                    item for item in other_answers if item[1] == unique_answer
                ]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    def eval_pred_list(self, pred_list, output_path = None):
        pred_scores = []
        for entry in tqdm(pred_list):
            pred_answer = self.answer_processor(entry["pred_ans"])
            unique_answer_scores = self._compute_answer_scores(entry["answers_gt"])
            score = unique_answer_scores.get(pred_answer, 0.0)
            # 记录 score 在 entry 中
            entry["score"] = score
            pred_scores.append(score)

        accuracy = sum(pred_scores) / len(pred_scores)

        # 保存结果
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(pred_list, f)

        return accuracy





# 主函数 
if __name__ == "__main__":
    output_file = '/mnt/workspace/zwq_data/interleaved_evaluation/TextVQA/TextVQA_pred_random_8_shot_basedmodel_hf_prompt.json'
     
    evaluator = TextVQAAccuracyEvaluator()
    with open(output_file, 'r') as f:
        pred_list = json.load(f)

    accuracy = evaluator.eval_pred_list(pred_list, output_path = output_file)

    print(f"Accuracy: {accuracy}")
    

