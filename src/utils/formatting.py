import re
import torch

class FormattingEvaluator:
    def __init__(self):
        self.SECOND_PATTERN = "Second, Verify that the decomposed elements align with the image."
        self.THIRD_PATTERN = "Third, Generate corrective feedback."

    # --- Helper methods ---
    def _split_text_into_parts(self, text):
        match_second = re.search(re.escape(self.SECOND_PATTERN), text)
        match_third = re.search(re.escape(self.THIRD_PATTERN), text)
        if match_second and match_third:
            part1 = text[:match_second.start()].strip()
            part2 = text[match_second.start():match_third.start()].strip().replace(self.SECOND_PATTERN, "").strip()
            part3 = text[match_third.start():].strip().replace(self.THIRD_PATTERN, "").strip()
            return part1, part2, part3
        return None, None, None

    def _parse_part1(self, text_block):
        if not text_block: return []
        lines = [line for line in text_block.split('\n') if line.strip()]
        parsed_lines = []
        for line in lines:
            try:
                num, contents = line.split('|', 1)
                parsed_lines.append((int(num.strip()), contents.strip()))
            except ValueError:
                continue
        return parsed_lines

    def _extract_answer_paragraphs(self, text_block):
        if not text_block: return []
        pattern = r'.*?Answer: (?:Yes|No)'
        paragraphs = re.findall(pattern, text_block, re.DOTALL)
        return [p.strip() for p in paragraphs]

    def _get_answer_from_paragraph(self, paragraph):
        match = re.search(r'Answer: (Yes|No)$', paragraph)
        return match.group(1) if match else None

    def _calculate_metrics(self, gt_part1, pred_part1, gt_answers, pred_paragraphs):
        """
        Calculate accuracy and format score (0/1).
        """
        metrics = {}

        # 1. [Accuracy] Part 1 content accuracy
        gt_contents = {content for _, content in gt_part1}
        pred_contents = {content for _, content in pred_part1}
        correct_matches = len(gt_contents.intersection(pred_contents))
        total_gt = len(gt_part1)
        metrics['part1_accuracy'] = correct_matches / total_gt if total_gt > 0 else 0.0

        # 2. [Accuracy] Part 2 answer accuracy (denominator: total GT count)
        correct_answers = 0
        pred_content_to_index = {content: i for i, (_, content) in enumerate(pred_part1)}
        for gt_index, (_, gt_content) in enumerate(gt_part1):
            if gt_content in pred_content_to_index:
                pred_para_index = pred_content_to_index[gt_content]
                if pred_para_index < len(pred_paragraphs) and gt_index < len(gt_answers):
                    gt_ans = self._get_answer_from_paragraph(gt_answers[gt_index])
                    pred_ans = self._get_answer_from_paragraph(pred_paragraphs[pred_para_index])
                    if gt_ans and gt_ans == pred_ans:
                        correct_answers += 1
        metrics['part2_accuracy'] = correct_answers / total_gt if total_gt > 0 else 0.0
        metrics['part2_accuracy_only_matching'] = correct_answers / correct_matches if correct_matches > 0 else 0.0

        # 3. [0/1 Score] Internal consistency: Pred Part 1 and Pred Part 2 item count match
        metrics['internal_consistency_ok'] = 1 if len(pred_part1) == len(pred_paragraphs) else 0

        # 4. [0/1 Score] Generation completeness: GT Part 1 and Pred Part 1 item count match
        metrics['part1_length_match_ok'] = 1 if len(gt_part1) == len(pred_part1) else 0

        final_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

        return final_metrics

    def _calculate_metrics_for_reward(self, gt_part1, pred_part1, pred_paragraphs):
        """
        Calculate accuracy and format score (0/1).
        """
        metrics = {}

        # 1. [Accuracy] Part 1 content accuracy
        gt_contents = {content for _, content in gt_part1}
        pred_contents = {content for _, content in pred_part1}
        correct_matches = len(gt_contents.intersection(pred_contents))
        total_gt = len(gt_part1)
        metrics['part1_accuracy'] = correct_matches / total_gt if total_gt > 0 else 0.0

        # 3. [0/1 Score] Internal consistency: Pred Part 1 and Pred Part 2 item count match
        metrics['internal_consistency_ok'] = 1 if len(pred_part1) == len(pred_paragraphs) else 0

        # 4. [0/1 Score] Generation completeness: GT Part 1 and Pred Part 1 item count match
        metrics['part1_length_match_ok'] = 1 if len(gt_part1) == len(pred_part1) else 0

        final_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

        return final_metrics

import re
import torch

class FormattingEvaluatorV2:
    def __init__(self):
        # Pattern definitions for the 4-stage structure
        self.SECOND_PATTERN = "Second, Decompose summarize"
        self.THIRD_PATTERN = "Third, Verify that the decomposed elements align with the image."
        self.FOURTH_PATTERN = "Fourth, Generate corrective feedback."

    def _split_text_into_parts(self, text):
        """
        Split the full text into 4 stages.
        Part 1: Summarize
        Part 2: Decompose summarize (Tuples)
        Part 3: Verify (Answers)
        Part 4: Feedback
        """
        m2 = re.search(re.escape(self.SECOND_PATTERN), text)
        m3 = re.search(re.escape(self.THIRD_PATTERN), text)
        m4 = re.search(re.escape(self.FOURTH_PATTERN), text)

        if m2 and m3 and m4:
            # Stage 1: start to before Second
            part1 = text[:m2.start()].strip()
            # Stage 2: Second to before Third
            part2 = text[m2.start():m3.start()].strip().replace(self.SECOND_PATTERN, "").strip()
            # Stage 3: Third to before Fourth
            part3 = text[m3.start():m4.start()].strip().replace(self.THIRD_PATTERN, "").strip()
            # Stage 4: Fourth to end
            part4 = text[m4.start():].strip().replace(self.FOURTH_PATTERN, "").strip()
            return part1, part2, part3, part4

        return None, None, None, None

    def _parse_part1(self, text_block):
        """Parse Decompose (Tuple) block."""
        if not text_block: return []
        lines = [line for line in text_block.split('\n') if line.strip()]
        parsed_lines = []
        for line in lines:
            try:
                num, contents = line.split('|', 1)
                parsed_lines.append((int(num.strip()), contents.strip()))
            except (ValueError, IndexError):
                continue
        return parsed_lines

    def _extract_answer_paragraphs(self, text_block):
        """Extract Verify (Yes/No) paragraphs."""
        if not text_block: return []
        pattern = r'.*?Answer: (?:Yes|No)'
        paragraphs = re.findall(pattern, text_block, re.DOTALL)
        return [p.strip() for p in paragraphs]

    def _get_answer_from_paragraph(self, paragraph):
        match = re.search(r'Answer: (Yes|No)$', paragraph)
        return match.group(1) if match else None

    def _calculate_metrics(self, gt_part1, pred_part1, gt_answers, pred_paragraphs):
        """
        Compute core evaluation metrics.
        gt_part1/pred_part1: list of parsed tuples
        gt_answers/pred_paragraphs: list of question/answer paragraphs
        """
        metrics = {}

        # 1. Part 1 (Decompose) content accuracy
        gt_contents = {content for _, content in gt_part1}
        pred_contents = {content for _, content in pred_part1}
        correct_matches = len(gt_contents.intersection(pred_contents))
        total_gt = len(gt_part1)
        metrics['part1_accuracy'] = correct_matches / total_gt if total_gt > 0 else 0.0

        # 2. Part 2 (Verify) answer accuracy
        correct_answers = 0
        pred_content_to_index = {content: i for i, (_, content) in enumerate(pred_part1)}
        for gt_index, (_, gt_content) in enumerate(gt_part1):
            if gt_content in pred_content_to_index:
                pred_para_index = pred_content_to_index[gt_content]
                if pred_para_index < len(pred_paragraphs) and gt_index < len(gt_answers):
                    gt_ans = self._get_answer_from_paragraph(gt_answers[gt_index])
                    pred_ans = self._get_answer_from_paragraph(pred_paragraphs[pred_para_index])
                    if gt_ans and gt_ans == pred_ans:
                        correct_answers += 1

        metrics['part2_accuracy'] = correct_answers / total_gt if total_gt > 0 else 0.0
        metrics['part2_accuracy_only_matching'] = correct_answers / correct_matches if correct_matches > 0 else 0.0

        # 3. Format and consistency scores
        metrics['internal_consistency_ok'] = 1 if len(pred_part1) == len(pred_paragraphs) else 0
        metrics['part1_length_match_ok'] = 1 if len(gt_part1) == len(pred_part1) else 0

        return {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
