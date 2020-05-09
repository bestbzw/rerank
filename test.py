from NewFinalUtils import ARG
from AnswerSelectMethods import ConcatAnswerSelectMethod
import sys
# 先创建一个参数集
args = ARG(eval_batch_size=4, 
            device='cuda:0',
            #device='cpu',
			model_name="./23_4_top2_on3285_models/EP"+sys.argv[1],
			tokenizer_name="/data/package/chinese_roberta_wwm_large_ext_pytorch",
            topk=2,
            save_dir = './23_4_top2_on3285_models/EP'+sys.argv[1])
            

method = ConcatAnswerSelectMethod(args=args)

method.read_all_files(NBEST_FILE_PATH='../results/ensemble_test1/ensemble_v2-14_3-17_2-17_3-22_5-23_3-23_4-25_4-26_5-27_5-28_5.nbest.json',
                      DATA_FILE_PATH='/data/bzw/MRC/data/lic2020/dureader_robust-test1/test1_dealed.json',
                      evaluate=True)

method.build_dev_data()
# 在这里，loss不降
method.fit_eval(use_logit=False)
