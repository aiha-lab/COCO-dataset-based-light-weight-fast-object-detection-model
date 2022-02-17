import io
import contextlib

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOEvaluator(object):

    def __init__(self, ann_json="instances_val2017.json"):
        self.ann_json = ann_json
        if ann_json is not None:
            self.coco = COCO(self.ann_json)
            self.class_ids = sorted(self.coco.getCatIds())
        else:
            raise ValueError("Annotation is not given...")

    def evaluate(self, pred_json: str):
        cocoGt = self.coco
        cocoDt = cocoGt.loadRes(pred_json)

        info = "Evaluate with PyCOCOTools:\n"

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info += redirect_string.getvalue()

        ap50_95 = cocoEval.stats[0]  # float
        ap50 = cocoEval.stats[1]  # float

        return ap50_95, ap50, info

# if __name__ == '__main__':
#     e = COCOEvaluator("E:/coco2017/annotations/instances_val2017.json")
#     print(e.class_ids)
#     print(len(e.class_ids))
